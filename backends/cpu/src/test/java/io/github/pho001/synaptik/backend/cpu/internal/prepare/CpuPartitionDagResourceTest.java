package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CpuPartitionDagResourceTest {
    @TempDir Path root;

    @Test void twoWorkspaceUnitsUseFinalIndicesAndExactFirstUseBufferCarrierMapping() {
        var analysis = analysis();
        var plan = analysis.plan();
        var workspaceIds = analysis.requirements().stream()
                .filter(PreparationResourceRequirement.Workspace.class::isInstance)
                .map(PreparationResourceRequirement.Workspace.class::cast)
                .map(PreparationResourceRequirement.Workspace::requirementId).toList();
        assertAll(
                () -> assertEquals(2, plan.units().size()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(2),
                                new ValueId(1), new ValueId(3)),
                        plan.boundaryValues()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(2)),
                        plan.units().get(0).boundaryValues()),
                () -> assertEquals(List.of(new ValueId(1), new ValueId(3)),
                        plan.units().get(1).boundaryValues()),
                () -> assertEquals(List.of(CarrierAccess.FLOAT_ARRAY,
                                CarrierAccess.MEMORY_SEGMENT),
                        plan.units().get(0).carrierPattern()),
                () -> assertEquals(List.of(CarrierAccess.MEMORY_SEGMENT,
                                CarrierAccess.FLOAT_ARRAY),
                        plan.units().get(1).carrierPattern()),
                () -> assertEquals(List.of(0L, 1L), workspaceIds),
                () -> assertEquals(0L, plan.units().get(0).runtimeFacts()
                        .workspaceDeclaration().orElseThrow().requirementId()),
                () -> assertEquals(1L, plan.units().get(1).runtimeFacts()
                        .workspaceDeclaration().orElseThrow().requirementId()),
                () -> assertTrue(plan.units().stream().allMatch(unit ->
                        unit.runtimeFacts().materialization().isEmpty())),
                () -> assertEquals(6, analysis.requirements().size()));
    }

    @Test void exactAssignmentsRealizeTwoArtifactsInStableOrder() throws Exception {
        var analysis = analysis();
        var workers = new CpuWorkerGroup(2);
        Path artifacts = root.resolve("valid");
        try {
            var executable = (CpuPreparedPartitionExecutable) finalize(analysis,
                    Mutation.NONE, artifacts, Optional.of(workers));
            assertAll(
                    () -> assertEquals(2, executable.children().size()),
                    () -> assertEquals(List.of(List.of(), List.of()),
                            executable.dependencies()),
                    () -> assertEquals(2, executable.memoryPlan().workspaces().size()),
                    () -> assertEquals(analysis.plan().units().get(0).portablePlan()
                                    .specialization(),
                            executable.children().get(0).artifact().specialization()),
                    () -> assertEquals(analysis.plan().units().get(1).portablePlan()
                                    .specialization(),
                            executable.children().get(1).artifact().specialization()),
                    () -> assertEquals(1,
                            analysis.plan().units().get(0).selectedRangeCount()),
                    () -> assertEquals(2,
                            analysis.plan().units().get(1).selectedRangeCount()));
        } finally {
            workers.close();
        }
    }

    @Test void malformedOrMissingBuffersAndWorkspacesFailBeforeArtifactLookup() {
        var analysis = analysis();
        for (Mutation mutation : List.of(Mutation.MISSING_BUFFER, Mutation.EXTRA_BUFFER,
                Mutation.MISMATCHED_BUFFER, Mutation.MISSING_WORKSPACE,
                Mutation.EXTRA_WORKSPACE, Mutation.MISMATCHED_WORKSPACE)) {
            Path artifacts = root.resolve(mutation.name().toLowerCase());
            var workers = new CpuWorkerGroup(2);
            try {
                assertThrows(IllegalArgumentException.class, () ->
                        finalize(analysis, mutation, artifacts, Optional.of(workers)),
                        mutation.name());
                assertFalse(Files.exists(artifacts), mutation.name());
            } finally {
                workers.close();
            }
        }
    }

    @Test void missingUndersizedAndClosedLaterWorkersFailBeforeTheFirstArtifact() {
        var analysis = analysis();
        Path missing = root.resolve("missing-worker");
        assertAll(
                () -> assertEquals(1, analysis.plan().units().get(0).selectedRangeCount()),
                () -> assertEquals(2, analysis.plan().units().get(1).selectedRangeCount()),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        finalize(analysis, Mutation.NONE, missing, Optional.empty())),
                () -> assertFalse(Files.exists(missing)));

        var undersized = new CpuWorkerGroup(1);
        Path small = root.resolve("undersized-worker");
        try {
            assertThrows(IllegalArgumentException.class, () ->
                    finalize(analysis, Mutation.NONE, small, Optional.of(undersized)));
            assertFalse(Files.exists(small));
        } finally {
            undersized.close();
        }

        var closed = new CpuWorkerGroup(2);
        closed.close();
        Path unavailable = root.resolve("closed-worker");
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () ->
                        finalize(analysis, Mutation.NONE, unavailable, Optional.of(closed))),
                () -> assertFalse(Files.exists(unavailable)));
    }

    @Test void invalidLaterCarrierAndLaterWorkspaceAbortColdBindingBeforeAnyWrite() {
        var analysis = analysis();
        var workers = new CpuWorkerGroup(2);
        try {
            var executable = (CpuPreparedPartitionExecutable) finalize(analysis,
                    Mutation.NONE, root.resolve("cold-validation"), Optional.of(workers));
            try (var wrongCarrier = run(executable, true, false)) {
                assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(wrongCarrier.state()));
                assertAll(
                        () -> assertNativeZeros(wrongCarrier.firstOutput(), 4),
                        () -> assertNativeZeros(wrongCarrier.laterNativeOutput(), 8));
            }
            try (var shortWorkspace = run(executable, false, true)) {
                float[] unchanged = shortWorkspace.laterOutput().clone();
                assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(shortWorkspace.state()));
                assertAll(
                        () -> assertNativeZeros(shortWorkspace.firstOutput(), 4),
                        () -> assertArrayEquals(unchanged, shortWorkspace.laterOutput()));
            }
            try (var readOnlyOutput = run(executable, false, false, true)) {
                float[] unchanged = readOnlyOutput.laterOutput().clone();
                assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(readOnlyOutput.state()));
                assertAll(
                        () -> assertNativeZeros(readOnlyOutput.firstOutput(), 4),
                        () -> assertArrayEquals(unchanged, readOnlyOutput.laterOutput()));
            }
        } finally {
            workers.close();
        }
    }

    @Test void firstChildFailurePreventsTheAlreadyBoundLaterChildFromExecuting() {
        var analysis = analysis();
        var workers = new CpuWorkerGroup(2);
        try {
            var executable = (CpuPreparedPartitionExecutable) finalize(analysis,
                    Mutation.NONE, root.resolve("hot-failure"), Optional.of(workers));
            try (var run = run(executable, false, false)) {
                float[] unchanged = run.laterOutput().clone();
                var invocation = executable.bind(run.state());
                run.firstWorkspace().close();
                assertThrows(RuntimeException.class, invocation::execute);
                assertArrayEquals(unchanged, run.laterOutput());
            }
        } finally {
            workers.close();
        }
    }

    @Test void oneRangeThenTwoRangeChildrenCompleteThroughOneCompositeInvocation() {
        var analysis = analysis();
        var workers = new CpuWorkerGroup(2);
        try {
            var executable = (CpuPreparedPartitionExecutable) finalize(analysis,
                    Mutation.NONE, root.resolve("complete"), Optional.of(workers));
            try (var run = run(executable, false, false)) {
                executable.bind(run.state()).execute();
                assertAll(
                        () -> assertArrayEquals(new float[] {1, 2, 3, 4},
                                nativeFloats(run.firstOutput(), 4)),
                        () -> assertArrayEquals(new float[] {1, 3, 7, 8, 2, 4, 5, 6},
                                run.laterOutput()));
            }
        } finally {
            workers.close();
        }
    }

    private Object finalize(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            Mutation mutation, Path artifactRoot, Optional<CpuWorkerGroup> workers) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                var slot = new BufferSlot(buffers.size());
                long size = mutation == Mutation.MISMATCHED_BUFFER && buffers.isEmpty()
                        ? buffer.byteSize() - 1 : buffer.byteSize();
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot, size,
                        buffer.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot,
                        buffers.size() - 1));
            } else {
                var workspace = (PreparationResourceRequirement.Workspace) requirement;
                var slot = new WorkspaceSlot(workspaces.size());
                long size = mutation == Mutation.MISMATCHED_WORKSPACE && workspaces.isEmpty()
                        ? workspace.byteSize() + workspace.byteAlignment() : workspace.byteSize();
                workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, size,
                        workspace.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(workspace, slot,
                        workspaces.size() - 1));
            }
        }
        if (mutation == Mutation.MISSING_BUFFER) {
            assignments.removeIf(PreparationResourceAssignment.Buffer.class::isInstance);
        } else if (mutation == Mutation.EXTRA_BUFFER) {
            assignments.add(assignments.stream()
                    .filter(PreparationResourceAssignment.Buffer.class::isInstance)
                    .findFirst().orElseThrow());
        } else if (mutation == Mutation.MISSING_WORKSPACE) {
            assignments.removeLast();
        } else if (mutation == Mutation.EXTRA_WORKSPACE) {
            assignments.add(assignments.stream()
                    .filter(PreparationResourceAssignment.Workspace.class::isInstance)
                    .findFirst().orElseThrow());
        }
        return new CpuPartitionFinalizer(Optional.of(artifactRoot), workers).finalizePartition(
                new BackendPartitionFinalization<>(analysis,
                        new PreparedMemoryPlan(buffers, workspaces), assignments));
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis() {
        return new CpuPartitionPreparer().analyze(context());
    }

    private static RunFixture run(CpuPreparedPartitionExecutable executable,
            boolean wrongLaterCarrier, boolean shortLaterWorkspace) {
        return run(executable, wrongLaterCarrier, shortLaterWorkspace, false);
    }

    private static RunFixture run(CpuPreparedPartitionExecutable executable,
            boolean wrongLaterCarrier, boolean shortLaterWorkspace, boolean readOnlyLaterOutput) {
        float[] firstInput = {1, 2, 3, 4};
        var firstOutput = CpuNativeBuffer.allocate(DataType.FLOAT32, 16, 4);
        var laterInput = CpuNativeBuffer.allocate(DataType.FLOAT32, 32, 4);
        float[] input = {8, 3, 7, 1, 6, 2, 5, 4};
        for (int index = 0; index < input.length; index++) laterInput.segment().set(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), index * 4L,
                input[index]);
        float[] laterOutput = {91, 92, 93, 94, 95, 96, 97, 98};
        CpuNativeBuffer laterNative = wrongLaterCarrier
                ? CpuNativeBuffer.allocate(DataType.FLOAT32, 32, 4) : null;
        BufferRepresentation fourth = wrongLaterCarrier ? laterNative
                : readOnlyLaterOutput ? borrowReadOnly(laterOutput) : borrow(laterOutput);
        var bindings = List.of(
                List.of(new BufferRepresentationBinding(borrow(firstInput),
                        RunResourceOwnership.BORROWED)),
                List.of(new BufferRepresentationBinding(firstOutput,
                        RunResourceOwnership.RUN_OWNED)),
                List.of(new BufferRepresentationBinding(laterInput,
                        RunResourceOwnership.BORROWED)),
                List.of(new BufferRepresentationBinding(fourth,
                        wrongLaterCarrier ? RunResourceOwnership.RUN_OWNED
                                : RunResourceOwnership.BORROWED)));
        var firstEntry = executable.memoryPlan().workspaces().get(0);
        var secondEntry = executable.memoryPlan().workspaces().get(1);
        var firstWorkspace = CpuContiguousWorkspace.allocate(firstEntry.byteSize(),
                firstEntry.byteAlignment());
        long secondBytes = shortLaterWorkspace
                ? secondEntry.byteSize() - secondEntry.byteAlignment() : secondEntry.byteSize();
        var secondWorkspace = CpuContiguousWorkspace.allocate(secondBytes,
                secondEntry.byteAlignment());
        var state = new RunState(executable.memoryPlan(), bindings,
                List.of(firstWorkspace, secondWorkspace));
        return new RunFixture(state, laterInput, firstOutput, laterNative, laterOutput,
                firstWorkspace);
    }

    private static CpuBorrowedBuffer borrow(float[] values) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, values.length,
                MemorySegment.ofArray(values)));
    }

    private static CpuBorrowedBuffer borrowReadOnly(float[] values) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, values.length,
                MemorySegment.ofArray(values).asReadOnly()));
    }

    private static void assertNativeZeros(CpuNativeBuffer buffer, int count) {
        assertArrayEquals(new float[count], nativeFloats(buffer, count));
    }

    private static float[] nativeFloats(CpuNativeBuffer buffer, int count) {
        var result = new float[count];
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
        for (int index = 0; index < count; index++) {
            result[index] = buffer.segment().get(layout, index * 4L);
        }
        return result;
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context() {
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(OrderingKind.SORT,
                        new SortAttrs(1, false)), List.of(new ValueId(0)),
                        List.of(new ValueId(2))),
                new CompiledNode(new NodeId(1), new Operation(OrderingKind.SORT,
                        new SortAttrs(1, false)), List.of(new ValueId(1)),
                        List.of(new ValueId(3))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var descriptors = List.of(descriptor(Shape.of(1, 4)), descriptor(Shape.of(2, 4)),
                descriptor(Shape.of(1, 4)), descriptor(Shape.of(2, 4)));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            boolean input = index < 2;
            values.add(new GraphValue(id, descriptors.get(index)));
            memory.add(new LogicalMemoryRequirement(id, descriptors.get(index),
                    input ? Optional.empty() : Optional.of(partition),
                    input ? List.of(partition) : List.of(), !input));
        }
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                2, 2, 1);
        var inputs = new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY), config);
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private enum Mutation {
        NONE,
        MISSING_BUFFER,
        EXTRA_BUFFER,
        MISMATCHED_BUFFER,
        MISSING_WORKSPACE,
        EXTRA_WORKSPACE,
        MISMATCHED_WORKSPACE
    }

    private record RunFixture(RunState state, CpuNativeBuffer borrowedLaterInput,
            CpuNativeBuffer firstOutput, CpuNativeBuffer laterNativeOutput, float[] laterOutput,
            CpuContiguousWorkspace firstWorkspace) implements AutoCloseable {
        @Override public void close() {
            state.close();
            borrowedLaterInput.close();
        }
    }
}
