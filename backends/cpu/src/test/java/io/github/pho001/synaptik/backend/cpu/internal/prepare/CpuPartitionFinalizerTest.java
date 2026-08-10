package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.nio.file.Files;

public class CpuPartitionFinalizerTest {
    @TempDir Path root;

    @Test void finalizesOneArtifactAndOnePartitionRecipeAfterExactAssignment() {
        var executable = finalizeExecutable(Shape.of(4), Optional.of(root));
        assertAll(
                () -> assertEquals(4, executable.bufferSelectionCount()),
                () -> assertEquals(4, executable.memoryPlan().buffers().size()),
                () -> assertEquals(4, executable.binding().end()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void parallelFinalizationRequiresAnOpenSufficientWorkerGroup() {
        var context = CpuPartitionPreparerTest.context(Shape.of(16));
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1));
        var analysis = new CpuPartitionPreparer().analyze(new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                context.partition(), context.nodes(), context.values(), context.memoryRequirements(),
                context.constants(), inputs));
        assertThrows(IllegalArgumentException.class,
                () -> finalizeExecutable(analysis, Optional.empty()));
        var undersized = new CpuWorkerGroup(1);
        try {
            assertThrows(IllegalArgumentException.class, () -> finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(undersized)));
        } finally { undersized.close(); }
        var closed = new CpuWorkerGroup(2);
        closed.close();
        assertThrows(IllegalArgumentException.class, () -> finalizeExecutable(analysis,
                Optional.empty(), Optional.of(closed)));
    }

    @Test void resolvesMaterializationWorkspaceAndWorkersBeforeArtifactPersistence() {
        var shape = Shape.of(2, 3);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, 48, 1, 1);
        var parallel = new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1);
        var analysis = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), parallel, policy));
        Path artifactRoot = root.resolve("must-remain-absent");
        assertAll(
                () -> assertEquals(5, analysis.requirements().size()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isPresent()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> finalizeExecutable(analysis, Optional.of(artifactRoot))),
                () -> assertFalse(Files.exists(artifactRoot)));
    }

    @Test void finalizesMovementAsOneArtifactWithoutWorkspace() {
        var analysis = new CpuPartitionPreparer().analyze(
                CpuNonAffineMovementLoweringTest.context(
                        new Operation(TensorCompositionKind.CONCAT,
                                new CompositionAxisAttrs(0)),
                        List.of(0, 1, 0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(1))),
                        CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(5))));
        var executable = finalizeExecutable(analysis, Optional.of(root.resolve("movement")));
        assertAll(
                () -> assertEquals(3, executable.bufferSelectionCount()),
                () -> assertEquals(3, executable.accessBindings().size()),
                () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    public static CpuPreparedExecutable finalizeExecutable(Shape shape, Optional<Path> root) {
        return finalizeExecutable(CpuPartitionPreparerTest.analyze(shape), root);
    }

    public static CpuPreparedExecutable finalizeExecutable(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Optional<Path> root) {
        return finalizeExecutable(analysis, root, Optional.empty());
    }

    public static CpuPreparedExecutable finalizeExecutable(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Optional<Path> root,
            Optional<CpuWorkerGroup> workerGroup) {
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        var workspaceEntries = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        for (int i = 0; i < analysis.requirements().size(); i++) {
            var any = analysis.requirements().get(i);
            if (any instanceof PreparationResourceRequirement.Workspace requirement) {
                var slot = new WorkspaceSlot(workspaceEntries.size());
                workspaceEntries.add(new PreparedMemoryPlan.WorkspaceEntry(slot,
                        requirement.byteSize(), requirement.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(requirement, slot,
                        workspaceEntries.size() - 1));
                continue;
            }
            var requirement = (PreparationResourceRequirement.Buffer) any;
            var slot = new BufferSlot(i);
            entries.add(new PreparedMemoryPlan.BufferEntry(slot, requirement.byteSize(),
                    requirement.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(requirement, slot, i));
        }
        var memoryPlan = new PreparedMemoryPlan(entries, workspaceEntries);
        return (CpuPreparedExecutable) new CpuPartitionFinalizer(root, workerGroup).finalizePartition(
                new BackendPartitionFinalization<>(analysis, memoryPlan, assignments));
    }
}
