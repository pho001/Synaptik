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

    public static CpuPreparedExecutable finalizeExecutable(Shape shape, Optional<Path> root) {
        return finalizeExecutable(CpuPartitionPreparerTest.analyze(shape), root);
    }

    public static CpuPreparedExecutable finalizeExecutable(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Optional<Path> root) {
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
        return (CpuPreparedExecutable) new CpuPartitionFinalizer(root).finalizePartition(
                new BackendPartitionFinalization<>(analysis, memoryPlan, assignments));
    }
}
