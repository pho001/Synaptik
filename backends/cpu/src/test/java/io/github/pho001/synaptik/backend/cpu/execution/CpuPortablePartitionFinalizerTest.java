package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CpuPortablePartitionFinalizerTest {
    @TempDir Path artifactRoot;

    @Test
    void mapsAssignmentsBeforeArtifactWorkAndRetainsExactPlanArtifactAndHandle() {
        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var candidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD);
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
        var requirement = assertInstanceOf(
                io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Buffer.class,
                analysis.requirements().getFirst());
        var slot = new BufferSlot(7);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan,
                List.of(new PreparationResourceAssignment.Buffer(requirement, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            var finalizer = new CpuPortablePartitionFinalizer(artifactRoot, workers);
            assertSame(CpuCapabilityProvider.CPU_BACKEND_ID, finalizer.backendId());
            var executable = assertInstanceOf(CpuPortablePreparedExecutable.class,
                    finalizer.finalizePartition(finalization));
            assertAll(
                    () -> assertSame(memoryPlan, executable.memoryPlan()),
                    () -> assertSame(executable.generatedKernel().entryPoint(), executable.entryPoint()),
                    () -> assertSame(candidate.specialization(),
                            executable.generatedKernel().specialization()),
                    () -> assertEquals(new PreparedExecutable.BufferSelection(0, 0),
                            executable.bufferSelection(0)));
            var reused = assertInstanceOf(CpuPortablePreparedExecutable.class,
                    finalizer.finalizePartition(finalization));
            assertSame(executable.generatedKernel(), reused.generatedKernel());
        }
    }

    @Test
    void missingBufferAndMismatchedWorkspaceUsesFailBeforeArtifactAccess() {
        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var assigned = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var selected = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var selectedCandidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortablePartitionPreparerTest.specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(CpuPortablePartitionPreparerTest.argument(
                                CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(selected),
                List.of(new CpuPortableKernelCandidate.BufferUse(selected, 0)), List.of(),
                CpuPortablePartitionPreparerTest.emitter(),
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        var analysis = new BackendPartitionAnalysis<>(context.partition(),
                new CpuPortablePreparationPlan(CpuPortablePartitionPreparerTest.partition(
                        selectedCandidate),
                        context.backendInputs().parallelConfiguration()), List.of(assigned));
        var slot = new BufferSlot(1);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan,
                List.of(new PreparationResourceAssignment.Buffer(assigned, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            assertEquals("bufferUses[0] has no assigned buffer requirement",
                    assertThrows(IllegalArgumentException.class,
                            () -> new CpuPortablePartitionFinalizer(artifactRoot, workers)
                                    .finalizePartition(finalization)).getMessage());
        }
        assertFalse(java.nio.file.Files.exists(artifactRoot.resolve("generated-kernels")));

        var selectedWorkspace = new PreparationResourceRequirement.Workspace(4, 8, 8);
        var workspaceCandidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortablePartitionPreparerTest.specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of()),
                List.of(selectedWorkspace), List.of(),
                List.of(new CpuPortableKernelCandidate.WorkspaceUse(selectedWorkspace)),
                CpuPortablePartitionPreparerTest.emitter(),
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        var workspaceAnalysis = new BackendPartitionAnalysis<>(context.partition(),
                new CpuPortablePreparationPlan(CpuPortablePartitionPreparerTest.partition(
                        workspaceCandidate),
                        context.backendInputs().parallelConfiguration()), List.of(assigned));
        var workspaceFinalization = new BackendPartitionFinalization<>(workspaceAnalysis,
                memoryPlan, List.of(new PreparationResourceAssignment.Buffer(assigned, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            assertEquals("workspaceUses[0] has no assigned workspace requirement",
                    assertThrows(IllegalArgumentException.class,
                            () -> new CpuPortablePartitionFinalizer(artifactRoot, workers)
                                    .finalizePartition(workspaceFinalization)).getMessage());
        }
        assertFalse(java.nio.file.Files.exists(artifactRoot.resolve("generated-kernels")));
    }

    @Test
    void validatesExplicitRootWorkerStateAndWorkerCountBeforeArtifactAccess() {
        try (var workers = new CpuWorkerGroup(1)) {
            assertEquals("artifactRoot", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePartitionFinalizer(null, workers)).getMessage());
            assertEquals("workerGroup", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePartitionFinalizer(artifactRoot, null)).getMessage());
            var finalizer = new CpuPortablePartitionFinalizer(artifactRoot, workers);
            assertEquals("finalization", assertThrows(NullPointerException.class,
                    () -> finalizer.finalizePartition(null)).getMessage());

            var context = CpuPortablePartitionPreparerTest.context(
                    CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
            var candidate = CpuPortablePartitionPreparerTest.candidate(
                    CpuPortableExecutionMode.SCALAR_SINGLE_THREAD);
            var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                    CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
            var requirement = (io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Buffer)
                    analysis.requirements().getFirst();
            var slot = new BufferSlot(1);
            var plan = new PreparedMemoryPlan(
                    List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
            var finalization = new BackendPartitionFinalization<>(analysis, plan,
                    List.of(new PreparationResourceAssignment.Buffer(requirement, slot, 0)));
            assertEquals("worker group count does not match prepared parallel configuration",
                    assertThrows(IllegalArgumentException.class,
                            () -> new CpuPortablePartitionFinalizer(artifactRoot, workers)
                                    .finalizePartition(finalization)).getMessage());
            assertFalse(java.nio.file.Files.exists(artifactRoot.resolve("generated-kernels")));
        }
        var closed = new CpuWorkerGroup(1);
        closed.close();
        assertEquals("CPU worker group is closed", assertThrows(IllegalStateException.class,
                () -> new CpuPortablePartitionFinalizer(artifactRoot, closed)).getMessage());

        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var candidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD);
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
        var requirement = (PreparationResourceRequirement.Buffer) analysis.requirements().getFirst();
        var slot = new BufferSlot(2);
        var plan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, plan,
                List.of(new PreparationResourceAssignment.Buffer(requirement, slot, 0)));
        var closedAfterConstruction = new CpuWorkerGroup(2);
        var finalizer = new CpuPortablePartitionFinalizer(artifactRoot, closedAfterConstruction);
        closedAfterConstruction.close();
        assertEquals("CPU worker group is closed", assertThrows(IllegalStateException.class,
                () -> finalizer.finalizePartition(finalization)).getMessage());
        assertFalse(java.nio.file.Files.exists(artifactRoot.resolve("generated-kernels")));
    }

    @Test
    void rejectsWrongOwnerBeforeArtifactAccess() {
        var context = CpuPortablePartitionPreparerTest.context(
                new io.github.pho001.synaptik.backend.contract.BackendId("other"), List.of());
        var candidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD);
        var requirement = candidate.requirements().getFirst();
        var analysis = new BackendPartitionAnalysis<>(context.partition(),
                new CpuPortablePreparationPlan(CpuPortablePartitionPreparerTest.partition(candidate),
                        context.backendInputs().parallelConfiguration()), List.of(requirement));
        var slot = new BufferSlot(1);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan,
                List.of(new PreparationResourceAssignment.Buffer(
                        (PreparationResourceRequirement.Buffer) requirement, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            assertEquals("partition owner must be CPU", assertThrows(IllegalArgumentException.class,
                    () -> new CpuPortablePartitionFinalizer(artifactRoot, workers)
                            .finalizePartition(finalization)).getMessage());
        }
        assertFalse(java.nio.file.Files.exists(artifactRoot.resolve("generated-kernels")));
    }

    @Test
    void artifactMissThenHitEmitsOnceAndExecutablesStronglyRetainExactArtifact() {
        var emissions = new AtomicInteger();
        var countingEmitter = new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() {
                return CpuPortablePartitionPreparerTest.LOWERING;
            }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                emissions.incrementAndGet(); scalar.code().return_();
            }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                emissions.incrementAndGet(); vector.code().return_();
            }
        };
        var requirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var candidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortablePartitionPreparerTest.specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(CpuPortablePartitionPreparerTest.argument(
                                CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(requirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(requirement, 0)), List.of(),
                countingEmitter,
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
        var slot = new BufferSlot(1);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan,
                List.of(new PreparationResourceAssignment.Buffer(requirement, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            var first = (CpuPortablePreparedExecutable)
                    new CpuPortablePartitionFinalizer(artifactRoot, workers)
                            .finalizePartition(finalization);
            assertEquals(1, emissions.get());
            var retained = first.generatedKernel();
            var second = (CpuPortablePreparedExecutable)
                    new CpuPortablePartitionFinalizer(artifactRoot, workers)
                            .finalizePartition(finalization);
            assertAll(() -> assertEquals(1, emissions.get()),
                    () -> assertTrue(java.nio.file.Files.exists(
                            artifactRoot.resolve("generated-kernels"))),
                    () -> assertSame(retained, first.generatedKernel()),
                    () -> assertSame(retained, second.generatedKernel()),
                    () -> assertSame(retained.entryPoint(), second.entryPoint()),
                    () -> assertSame(candidate.specialization(), retained.specialization()));
        }
    }

    @Test
    void mapsRepeatedBufferUsesAndRepresentationIndicesWithoutReselection() {
        var requirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var candidate = CpuPortablePartitionPreparerTest.candidate(
                CpuPortablePartitionPreparerTest.specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(CpuPortablePartitionPreparerTest.argument(
                                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY),
                                CpuPortablePartitionPreparerTest.argument(
                                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(requirement), List.of(
                        new CpuPortableKernelCandidate.BufferUse(requirement, 0),
                        new CpuPortableKernelCandidate.BufferUse(requirement, 2)), List.of(),
                CpuPortablePartitionPreparerTest.emitter(),
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
        var slot = new BufferSlot(1);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(slot, 16, 4)), List.of());
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan,
                List.of(new PreparationResourceAssignment.Buffer(requirement, slot, 0)));
        try (var workers = new CpuWorkerGroup(2)) {
            var executable = (CpuPortablePreparedExecutable)
                    new CpuPortablePartitionFinalizer(artifactRoot, workers)
                            .finalizePartition(finalization);
            assertAll(() -> assertEquals(new PreparedExecutable.BufferSelection(0, 0),
                            executable.bufferSelection(0)),
                    () -> assertEquals(new PreparedExecutable.BufferSelection(0, 2),
                            executable.bufferSelection(1)),
                    () -> assertSame(candidate.specialization(),
                            executable.generatedKernel().specialization()));
        }
    }

    @Test
    void mapsRepeatedWorkspaceUseToOneAssignedDirectWorkspace() {
        var bufferRequirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var workspaceRequirement = new PreparationResourceRequirement.Workspace(0, 8, 8);
        var candidate = new CpuPortableKernelCandidate(
                CpuPortablePartitionPreparerTest.specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(CpuPortablePartitionPreparerTest.argument(
                                CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                CpuPortablePartitionPreparerTest.emitter(),
                List.of(bufferRequirement, workspaceRequirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(bufferRequirement, 0)),
                List.of(new CpuPortableKernelCandidate.WorkspaceUse(workspaceRequirement),
                        new CpuPortableKernelCandidate.WorkspaceUse(workspaceRequirement)),
                (state, handle, spec, parallel, workers, buffers, workspaces) ->
                        new WorkspaceInvocation(state, workspaces[0], workspaces[1]));
        var context = CpuPortablePartitionPreparerTest.context(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(
                CpuPortablePartitionPreparerTest.partition(candidate))).analyze(context);
        var bufferSlot = new BufferSlot(1);
        var workspaceSlot = new WorkspaceSlot(1);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(bufferSlot, 16, 4)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot, 8, 8)));
        var finalization = new BackendPartitionFinalization<>(analysis, memoryPlan, List.of(
                new PreparationResourceAssignment.Buffer(bufferRequirement, bufferSlot, 0),
                new PreparationResourceAssignment.Workspace(workspaceRequirement, workspaceSlot, 0)));
        var buffer = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
        var workspace = CpuNativeWorkspace.allocate(8, 8);
        var state = new RunState(memoryPlan,
                List.of(List.of(new BufferRepresentationBinding(
                        buffer, RunResourceOwnership.BORROWED))), List.of(workspace));
        try (var workers = new CpuWorkerGroup(2)) {
            var executable = new CpuPortablePartitionFinalizer(artifactRoot, workers)
                    .finalizePartition(finalization);
            var invocation = assertInstanceOf(WorkspaceInvocation.class, executable.bind(state));
            assertSame(workspace, invocation.first);
            assertSame(invocation.first, invocation.second);
            invocation.execute();
            assertEquals(1, invocation.calls);
        } finally {
            state.close();
        }
    }

    private static final class WorkspaceInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        final CpuNativeWorkspace first;
        final CpuNativeWorkspace second;
        int calls;
        WorkspaceInvocation(RunState state, CpuNativeWorkspace first, CpuNativeWorkspace second) {
            super(state); this.first = first; this.second = second;
        }
        @Override protected void executeBound() { calls++; }
    }
}
