package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Bounded profile-free CPU capability, preparation, and represented-value execution. */
final class CpuProfileFreeConformanceTest {
    private static final Shape SHAPE = Shape.of(3);
    private static final TensorDescriptor FLOAT = descriptor(DataType.FLOAT32);

    @Test
    void supportedNegPreparesAndPreservesRepresentedExecution() {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            var operation = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
            var query = new OperationCapabilityQuery(
                    operation, List.of(FLOAT), List.of(FLOAT));
            var bool = descriptor(DataType.BOOL);
            assertTrue(integration.capabilityProvider().supports(query));
            assertFalse(integration.capabilityProvider().supports(new OperationCapabilityQuery(
                    operation, List.of(bool), List.of(bool))));

            Evidence result = execute(integration);
            assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE, result.route());
            assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR,
                    result.strategy().compute());
            assertEquals(1, result.rangeCount());
            assertTrue(result.minimumElementsPerWorker() > 0);
            assertTrue(result.vectorSpeciesBits() >= 0);
            assertFalse(result.irKey().isBlank());
            assertFalse(result.specializationKey().isBlank());
            assertArrayEquals(new float[] {-1.25f, 2.5f, -4.0f}, result.output());
        }
    }

    private static Evidence execute(CpuBackendIntegration integration) {
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                new CpuPartitionPreparer().analyze(context());
        CpuPartitionPreparationPlan plan = analysis.plan();
        PreparedExecutable executable = finalizePortable(analysis);
        float[] input = {1.25f, -2.5f, 4.0f};
        float[] output = new float[input.length];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputSegment = copy(arena, input);
            MemorySegment outputSegment = copy(arena, output);
            try (RunState state = new RunState(executable.memoryPlan(), List.of(
                    List.of(new BufferRepresentationBinding(integration.borrow(
                            new MemorySegmentStorage(DataType.FLOAT32, input.length, inputSegment)),
                            RunResourceOwnership.BORROWED)),
                    List.of(new BufferRepresentationBinding(integration.borrow(
                            new MemorySegmentStorage(DataType.FLOAT32, output.length, outputSegment)),
                            RunResourceOwnership.BORROWED))), List.of())) {
                executable.bind(state).execute();
            }
            output = outputSegment.toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
        }
        var unit = plan.units().getFirst();
        return new Evidence(plan.route(), unit.executionStrategy(),
                unit.selectedRangeCount(), unit.minimumElementsPerWorker(),
                unit.vectorSpeciesBitSize(), unit.portablePlan().kernelIr().structuralKey(),
                unit.portablePlan().specialization().structuralKey(), output);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context() {
        var input = new ValueId(0);
        var output = new ValueId(1);
        var node = new CompiledNode(new NodeId(0),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(input), List.of(output));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var values = List.of(new GraphValue(input, FLOAT), new GraphValue(output, FLOAT));
        var memory = List.of(
                new LogicalMemoryRequirement(input, FLOAT, Optional.empty(),
                        List.of(partition), false),
                new LogicalMemoryRequirement(output, FLOAT, Optional.of(partition),
                        List.of(), true));
        var inputs = new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT),
                new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1));
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                inputs);
    }

    private static PreparedExecutable finalizePortable(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                var slot = new BufferSlot(buffers.size());
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                        buffer.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(
                        buffer, slot, buffers.size() - 1, 0));
            } else if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                var slot = new WorkspaceSlot(workspaces.size());
                workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, workspace.byteSize(),
                        workspace.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(
                        workspace, slot, workspaces.size() - 1));
            }
        }
        var memory = new PreparedMemoryPlan(buffers, workspaces);
        return new CpuPartitionFinalizer().finalizePartition(
                new BackendPartitionFinalization<>(analysis, memory, assignments)).executable();
    }

    private static MemorySegment copy(Arena arena, float[] values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment target = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, target, 0, source.byteSize());
        return target;
    }

    private static TensorDescriptor descriptor(DataType type) {
        return new TensorDescriptor(type, SHAPE,
                Optional.of(LayoutDescriptor.contiguous(SHAPE)), false);
    }

    private record Evidence(CpuPartitionPreparationPlan.Route route,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy, int rangeCount,
            long minimumElementsPerWorker, int vectorSpeciesBits, String irKey,
            String specializationKey, float[] output) {
        private Evidence {
            output = output.clone();
        }
    }
}
