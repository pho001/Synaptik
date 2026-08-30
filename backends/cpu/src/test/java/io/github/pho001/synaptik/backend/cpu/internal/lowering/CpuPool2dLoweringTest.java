package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public class CpuPool2dLoweringTest {
    @Test
    void lowersLiteralCeilTailAndZeroBatch() {
        var geometry =
                new CpuPool2dLowering()
                        .lower(
                                context(
                                        Pool2dKind.MAX_POOL2D,
                                        new io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs(
                                                2, 2, 2, 2, 2, 2, 1, 1, true),
                                        DataType.FLOAT32,
                                        Shape.of(1, 2, 3, 3),
                                        Shape.of(1, 2, 4, 4)))
                        .pool2dGeometry()
                        .orElseThrow();
        assertEquals(32, geometry.outputCount());
        assertEquals(
                0,
                new CpuPool2dLowering()
                        .lower(
                                context(
                                        Pool2dKind.AVERAGE_POOL2D,
                                        new io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs(
                                                1, 1, 1, 1, 0, 0, 1, 1, false),
                                        DataType.FLOAT64,
                                        Shape.of(0, 1, 2, 2),
                                        Shape.of(0, 1, 2, 2)))
                        .elementCount());
    }

    @Test
    void rejectsMismatchedOutputAndNonInjectiveOutput() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new CpuPool2dLowering()
                                .lower(
                                        context(
                                                Pool2dKind.MAX_POOL2D,
                                                new io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs(
                                                        1, 1, 1, 1, 0, 0, 1, 1, false),
                                                DataType.FLOAT32,
                                                Shape.of(1, 1, 2, 2),
                                                Shape.of(1, 1, 1, 1))));
    }

    @Test
    void preparationRetainsSchema55GeometryAndNoWorkspace() {
        var plan =
                new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                        .analyze(
                                context(
                                        Pool2dKind.AVERAGE_POOL2D,
                                        new io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs(
                                                2, 2, 1, 1, 0, 0, 1, 1, false),
                                        DataType.FLOAT64,
                                        Shape.of(1, 1, 3, 3),
                                        Shape.of(1, 1, 2, 2)))
                        .plan();
        assertAll(
                () ->
                        assertEquals(
                                55, plan.units().getFirst().portablePlan().specialization().classIdentitySchema()),
                () -> assertTrue(plan.units().getFirst().pool2dGeometry().isPresent()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertEquals(4, plan.elementCount()));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(
            Pool2dKind kind, OperationAttrs attrs, DataType type, Shape inputShape, Shape outputShape) {
        var input =
                new TensorDescriptor(
                        type, inputShape, Optional.of(LayoutDescriptor.contiguous(inputShape)), false);
        var output =
                new TensorDescriptor(
                        type, outputShape, Optional.of(LayoutDescriptor.contiguous(outputShape)), false);
        return CpuScatterLoweringTest.context(
                new Operation(kind, attrs), List.of(0), List.of(input), output);
    }
}
