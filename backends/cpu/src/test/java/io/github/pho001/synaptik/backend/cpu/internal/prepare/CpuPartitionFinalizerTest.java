package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
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
        for (int i = 0; i < analysis.requirements().size(); i++) {
            var requirement = (PreparationResourceRequirement.Buffer) analysis.requirements().get(i);
            var slot = new BufferSlot(i);
            entries.add(new PreparedMemoryPlan.BufferEntry(slot, requirement.byteSize(),
                    requirement.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(requirement, slot, i));
        }
        var memoryPlan = new PreparedMemoryPlan(entries, List.of());
        return (CpuPreparedExecutable) new CpuPartitionFinalizer(root, workerGroup).finalizePartition(
                new BackendPartitionFinalization<>(analysis, memoryPlan, assignments));
    }
}
