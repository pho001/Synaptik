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
    @Test void selectsFirstLowestEligibleCopyAndDeclaresExactWorkspace() {
        Shape shape = Shape.of(2, 3);
        var general = descriptor(shape, LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        var dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var policy = new MaterializationPolicy(true, 0, 1, 10, 1, 2,
                48, 1, 1);
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy);
        var analysis = CpuPartitionPreparerTest.analyze(general, general, dense, dense, inputs);
        var copy = analysis.plan().materialization().orElseThrow();
        assertAll(
                () -> assertEquals(0, copy.sourceBoundaryIndex()),
                () -> assertEquals(48, copy.byteCount()),
                () -> assertEquals(1, copy.useCount()),
                () -> assertEquals(120, copy.directCost()),
                () -> assertEquals(24, copy.copiedTotalCost()),
                () -> assertEquals(5, analysis.requirements().size()),
                () -> assertEquals(0, analysis.plan().workspaceDeclaration().orElseThrow().requirementId()),
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
                .plan().materialization().isEmpty());
        var overflow = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                new MaterializationPolicy(true, Long.MAX_VALUE, Long.MAX_VALUE,
                        Long.MAX_VALUE, 0, 2, Long.MAX_VALUE, 0, 0));
        assertThrows(ArithmeticException.class, () -> new CpuPartitionPreparer().analyze(
                CpuPartitionPreparerTest.context(general, dense, dense, dense, overflow)));
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
