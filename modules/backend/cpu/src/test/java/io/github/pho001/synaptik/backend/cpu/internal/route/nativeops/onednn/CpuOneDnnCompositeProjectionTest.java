package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Binary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.BinaryAlgorithm;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Pipeline;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Pooling;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Reduction;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.ReductionAlgorithm;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Unary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.UnaryAlgorithm;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Behavioral checks of complete common-unit recipes; no native library or execution is needed. */
final class CpuOneDnnCompositeProjectionTest {
    @Test void mseRetainsRepeatedRolesAndMandatoryMeanDivision() {
        for (var reduction : LossReduction.values()) {
            var shape = Shape.of(2, 3);
            var input = descriptor(DataType.FLOAT32, shape, new long[] {1, 2}, 7);
            var output = reduction == LossReduction.NONE
                    ? descriptor(DataType.FLOAT32, shape, new long[] {3, 1}, 4)
                    : descriptor(DataType.FLOAT32, Shape.scalar(), new long[0], 4);
            var unit = unit(CpuScatterLoweringTest.context(
                    new Operation(LossKind.MEAN_SQUARED_ERROR, new MeanSquaredErrorAttrs(reduction)),
                    List.of(0, 0), List.of(input), output));
            var projection = CpuOneDnnCompositeProjection.from(unit).orElseThrow();
            var pipeline = assertInstanceOf(Pipeline.class, projection.spec());
            assertEquals(List.of(0), projection.inputBoundaryPositions());
            assertEquals(1, projection.outputBoundaryPosition());
            assertEquals(List.of(28L, 16L), projection.byteOffsets());
            assertEquals(List.of(0, 0), pipeline.steps().getFirst().inputSlots());
            assertEquals(BinaryAlgorithm.SUB, ((Binary) pipeline.steps().getFirst().spec()).algorithm());
            assertEquals(BinaryAlgorithm.MUL, ((Binary) pipeline.steps().get(1).spec()).algorithm());
            assertEquals(1, pipeline.steps().getLast().outputSlot());
            if (reduction == LossReduction.NONE) {
                assertEquals(2, pipeline.steps().size());
                assertTrue(pipeline.constants().isEmpty());
            } else {
                var sum = assertInstanceOf(Reduction.class, pipeline.steps().get(2).spec());
                assertEquals(ReductionAlgorithm.SUM, sum.algorithm());
                assertEquals(List.of(1L, 1L), sum.output().dimensions());
                if (reduction == LossReduction.MEAN) {
                    assertEquals(4, pipeline.steps().size());
                    assertEquals(BinaryAlgorithm.DIV, ((Binary) pipeline.steps().getLast().spec()).algorithm());
                    assertEquals(List.of(Float.floatToRawIntBits(6.0f)),
                            pipeline.constants().getFirst().rawWords());
                } else {
                    assertEquals(3, pipeline.steps().size());
                    assertTrue(pipeline.constants().isEmpty());
                }
            }
        }
    }

    @Test void batchNormPreservesAllSixSitesAndSharedChannelRoles() {
        var shape = Shape.of(2, 3, 4);
        var vector = Shape.of(3);
        var unit = unit(CpuScatterLoweringTest.context(
                new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                        new BatchNormInferenceAttrs(1, ScalarValue.float32(1e-5f))),
                List.of(0, 1, 1, 1, 1),
                List.of(descriptor(DataType.FLOAT32, shape, new long[] {12, 4, 1}, 5),
                        descriptor(DataType.FLOAT32, vector, new long[] {2}, 7)),
                descriptor(DataType.FLOAT32, shape, new long[] {12, 4, 1}, 9)));
        var projection = CpuOneDnnCompositeProjection.from(unit).orElseThrow();
        var pipeline = assertInstanceOf(Pipeline.class, projection.spec());
        assertEquals(List.of(0, 1), projection.inputBoundaryPositions());
        assertEquals(List.of(20L, 28L, 36L), projection.byteOffsets());
        assertEquals(6, pipeline.steps().size());
        assertEquals(BinaryAlgorithm.SUB, ((Binary) pipeline.steps().get(0).spec()).algorithm());
        assertEquals(BinaryAlgorithm.ADD, ((Binary) pipeline.steps().get(1).spec()).algorithm());
        assertEquals(UnaryAlgorithm.SQRT, ((Unary) pipeline.steps().get(2).spec()).algorithm());
        assertEquals(BinaryAlgorithm.DIV, ((Binary) pipeline.steps().get(3).spec()).algorithm());
        assertEquals(BinaryAlgorithm.MUL, ((Binary) pipeline.steps().get(4).spec()).algorithm());
        assertEquals(BinaryAlgorithm.ADD, ((Binary) pipeline.steps().get(5).spec()).algorithm());
        var centered = (Binary) pipeline.steps().getFirst().spec();
        assertEquals(List.of(1L, 3L, 1L), centered.right().dimensions());
        assertEquals(List.of(0L, 2L, 0L), centered.right().strides());
        assertEquals(List.of(Float.floatToRawIntBits(1e-5f)), pipeline.constants().getFirst().rawWords());
        assertEquals(2, pipeline.steps().getLast().outputSlot());
    }

    @Test void poolingRequiresPaddingInEveryWindowNotMerelyAtTheEdges() {
        assertTrue(CpuOneDnnCompositeProjection.from(pool2d(Shape.of(1, 1, 4, 4),
                Shape.of(1, 1, 5, 5), new AveragePool2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false))).isEmpty());
        assertTrue(CpuOneDnnCompositeProjection.from(pool2d(Shape.of(1, 1, 1, 1),
                Shape.of(1, 1, 1, 1), new AveragePool2dAttrs(3, 3, 1, 1, 1, 1, 1, 1, false))).isEmpty());
        var projection = CpuOneDnnCompositeProjection.from(pool2d(Shape.of(1, 1, 1, 1),
                Shape.of(1, 1, 2, 2), new AveragePool2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false)))
                .orElseThrow();
        var pooling = assertInstanceOf(Pooling.class, projection.spec());
        assertEquals(List.of(2L, 2L), pooling.kernel());
        assertEquals(List.of(1L, 1L), pooling.paddingLeft());
        assertEquals(List.of(1L, 1L), pooling.paddingRight());
    }

    @Test void poolingPreservesLiteralCeilGridAndDilatedThreeDimensionalSamples() {
        var ceil = (Pooling) CpuOneDnnCompositeProjection.from(pool2d(Shape.of(1, 1, 1, 2),
                Shape.of(1, 1, 2, 2), new AveragePool2dAttrs(2, 2, 2, 2, 1, 1, 1, 1, true)))
                .orElseThrow().spec();
        assertEquals(List.of(1L, 1L, 2L, 2L), ceil.output().dimensions());
        assertEquals(List.of(2L, 1L), ceil.paddingRight());
        var input = descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1));
        var output = descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1));
        var unit = unit(CpuScatterLoweringTest.context(new Operation(Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, false)),
                List.of(0), List.of(input), output));
        var pooling = (Pooling) CpuOneDnnCompositeProjection.from(unit).orElseThrow().spec();
        assertEquals(List.of(2L, 2L, 2L), pooling.dilations());
        assertEquals(List.of(1L, 1L, 1L), pooling.paddingLeft());
    }

    @Test void unsupportedTypesAndEmptyDomainsStayOutsideNativeProjection() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var shape = type == DataType.FLOAT32 ? Shape.of(0) : Shape.of(3);
            var input = descriptor(type, shape);
            var unit = unit(CpuScatterLoweringTest.context(new Operation(LossKind.MEAN_SQUARED_ERROR,
                    new MeanSquaredErrorAttrs(LossReduction.NONE)), List.of(0, 0),
                    List.of(input), input));
            assertTrue(CpuOneDnnCompositeProjection.from(unit).isEmpty());
        }
    }

    private static CpuPartitionPreparationPlan.ExecutionUnitPlan pool2d(
            Shape input, Shape output, AveragePool2dAttrs attrs) {
        return unit(CpuScatterLoweringTest.context(new Operation(Pool2dKind.AVERAGE_POOL2D, attrs),
                List.of(0), List.of(descriptor(DataType.FLOAT32, input)),
                descriptor(DataType.FLOAT32, output)));
    }

    private static CpuPartitionPreparationPlan.ExecutionUnitPlan unit(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, long[] strides, long offset) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }
}
