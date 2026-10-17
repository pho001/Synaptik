package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.MaterializationPolicy;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuMaterializationPlanTest {
    @Test void retainsFirstLowestEligibleCopyAndKeepsOrdinarySelectionDirect() {
        Shape shape = Shape.of(2, 3);
        var general = descriptor(shape, LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        var dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var policy = new MaterializationPolicy(true, 0, 1, 10, 1, 2,
                48, 1, 1);
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy);
        var analysis = CpuPartitionPreparerTest.analyze(general, general, dense, dense, inputs);
        var variant = analysis.plan().representationDecisions().stream()
                .filter(io.github.pho001.synaptik.backend.cpu.internal.ir
                        .CpuRepresentationDecision.Variant.class::isInstance)
                .map(io.github.pho001.synaptik.backend.cpu.internal.ir
                        .CpuRepresentationDecision.Variant.class::cast)
                .filter(value -> value.identity().materializations().stream().map(
                        io.github.pho001.synaptik.backend.cpu.internal.ir
                                .CpuRepresentationDecision.MaterializationIdentity
                                ::sourceBoundaryPosition).toList().equals(java.util.List.of(0)))
                .findFirst().orElseThrow();
        var copy = variant.identity().materializations().getFirst();
        var selection = (io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuRepresentationDecision.Selection) analysis.plan()
                .representationDecisions().getLast();
        assertAll(
                () -> assertTrue(analysis.plan().materializations().isEmpty()),
                () -> assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir
                                .CpuRepresentationDecision.SelectionReason
                                .DIRECT_MATERIALIZATION_UNPROVED,
                        selection.reason()),
                () -> assertEquals(0, copy.sourceBoundaryPosition()),
                () -> assertEquals(48, copy.byteCount()),
                () -> assertEquals(1, copy.instructionUseCount()),
                () -> assertEquals(120, variant.selectedDirectCost().orElseThrow()),
                () -> assertEquals(24, variant.selectedCopiedCost().orElseThrow()),
                () -> assertEquals(4, analysis.requirements().size()),
                () -> assertEquals(8, copy.workspaceRequirementId()),
                () -> assertEquals(48, copy.workspaceBytes()),
                () -> assertEquals(4, analysis.plan().specializationBudget().candidatePlans()));
    }

    @Test void directWinsTiesAndHardFiltersDenseScalarAndMemoryLimit() {
        Shape shape = Shape.of(2, 3);
        var general = descriptor(shape, LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        var dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var tie = new MaterializationPolicy(true, 0, 1, 1, 0, 1, 48, 0, 0);
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, tie);
        assertTrue(CpuPartitionPreparerTest.analyze(general, dense, dense, dense, inputs)
                .plan().materializations().isEmpty());
        var overflow = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                new MaterializationPolicy(true, Long.MAX_VALUE, Long.MAX_VALUE,
                        Long.MAX_VALUE, 0, 2, Long.MAX_VALUE, 0, 0));
        assertTrue(new CpuPartitionPreparer().analyze(
                CpuPartitionPreparerTest.context(general, dense, dense, dense, overflow))
                .plan().materializations().isEmpty());
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
