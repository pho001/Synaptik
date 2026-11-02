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

    /**
     * Proves the unsupported outcome of a numerical-family low fixture without removing its
     * dtype/carrier/layout row. Exact/storage-only families must not call this helper.
     * @param context exact independently constructed numerical-family fixture
     * @return whether both actual source and independent preparation rejected a low row
     */
    static boolean assertLockedNumerical(PrepareContext<CpuPartitionAnalysisInputs> context) {
        if (context.values().stream().noneMatch(value -> value.descriptor().dataType() == DataType.BFLOAT16
                || value.descriptor().dataType() == DataType.FLOAT16)) return false;
        var descriptors = new java.util.HashMap<ValueId, TensorDescriptor>();
        context.values().forEach(value -> descriptors.put(value.id(), value.descriptor()));
        boolean rejected = false;
        for (var node : context.nodes()) {
            var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(descriptors::get).toList(),
                    node.outputs().stream().map(descriptors::get).toList(), node.scanBody(),
                    Optional.ofNullable(context.scanActivityPlans().get(node.id())));
            rejected |= !new CpuCapabilityProvider().supports(query);
        }
        assertTrue(rejected, "a low numerical source occurrence must have no CPU owner");
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(context));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().completePlanCandidates(context));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering().lower(context));
        return true;
    }

    /** Exact fusion keeps producing-site rounding, fresh run state and all-input alias checks. */
    @Test void ordinaryProducerExactSumTailPreservesRoundingResourcesAndFailBeforeWrites() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var shapes = List.of(Shape.of(2),Shape.of(2),Shape.of(2),Shape.scalar(),Shape.scalar(),Shape.scalar());
            var nodes = List.of(new CompiledNode(new NodeId(0),new Operation(
                    io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.MUL,NoOperationAttrs.INSTANCE),
                    List.of(new ValueId(0),new ValueId(1)),List.of(new ValueId(2))),
                    new CompiledNode(new NodeId(1),new Operation(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                            NoOperationAttrs.INSTANCE),List.of(new ValueId(2)),List.of(new ValueId(3))),
                    new CompiledNode(new NodeId(2),new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,
                            NoOperationAttrs.INSTANCE),List.of(new ValueId(3),new ValueId(4)),List.of(new ValueId(5))));
            var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
            var values = new ArrayList<GraphValue>(); var requirements = new ArrayList<LogicalMemoryRequirement>();
            for (int i = 0; i < shapes.size(); i++) {
                var id = new ValueId(i); var desc = new TensorDescriptor(type,shapes.get(i),Optional.of(LayoutDescriptor.contiguous(shapes.get(i))),false);
                values.add(new GraphValue(id,desc));
                requirements.add(new LogicalMemoryRequirement(id,desc,i == 2 || i == 3 || i == 5 ? Optional.of(partition) : Optional.empty(),
                        i == 5 ? List.of() : List.of(partition),i == 5));
            }
            var analysis = new CpuPartitionPreparer().analyze(new PrepareContext<>(partition,nodes,values,requirements,Map.of(),CpuPartitionAnalysisInputs.DEFAULT));
            assertEquals(1,analysis.plan().units().size()); assertEquals(4,analysis.plan().bufferDeclarations().size());
            assertEquals(type == DataType.FLOAT32 ? 48 : 272,analysis.plan().aggregateGeometry().orElseThrow().workspaceBytes(1));
            var executable = finalizePortable(analysis);
            try (var integration = CpuBackendIntegration.open(); var arena = Arena.ofConfined()) {
                for (int run = 0; run < 2; run++) for (int alias = -1; alias < 3; alias++) {
                    int width = type.byteWidth(); double epsilon = Math.scalb(1d,type == DataType.FLOAT32 ? -23 : -52);
                    var left = arena.allocate(2L*width,width); var right = arena.allocate(2L*width,width); var bias = arena.allocate(width,width);
                    for (int i = 0; i < 2; i++) {
                        double a = i == 0 ? 1+epsilon : -(1+2*epsilon), b = i == 0 ? 1+epsilon : 1;
                        if (type == DataType.FLOAT32) { left.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i,(float)a); right.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i,(float)b); }
                        else { left.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,i,a); right.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,i,b); }
                    }
                    if (type == DataType.FLOAT32) bias.set(java.lang.foreign.ValueLayout.JAVA_FLOAT,0,run+1);
                    else bias.set(java.lang.foreign.ValueLayout.JAVA_DOUBLE,0,run+1);
                    var reads = List.of(left,right,bias);
                    var output = alias < 0 ? arena.allocate(width,width) : reads.get(alias).asSlice(alias < 2 ? width : 0,width);
                    byte[] unchanged = output.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                    var bindings = new ArrayList<List<BufferRepresentationBinding>>();
                    for (var segment : List.of(left,right,bias,output)) bindings.add(List.of(new BufferRepresentationBinding(integration.borrow(
                            new MemorySegmentStorage(type,segment.byteSize()/width,segment)),RunResourceOwnership.BORROWED)));
                    var workspace = io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace.allocate(
                            type == DataType.FLOAT32 ? 48 : 272,8);
                    try (var state = new RunState(executable.memoryPlan(),bindings,List.of(workspace))) {
                        if (alias >= 0) {
                            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,() -> executable.bind(state));
                            assertArrayEquals(unchanged,output.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                        } else {
                            executable.bind(state).execute();
                            assertEquals(run+1,type == DataType.FLOAT32 ? output.get(java.lang.foreign.ValueLayout.JAVA_FLOAT,0)
                                    : output.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE,0));
                        }
                    }
                }
            }
        }
    }

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
