package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuFoldLoweringTest {
    @Test void lowersAxisAndNchwGeometryWithExactlyTwoBoundariesAndNoOtherGeometry() {
        var axis = lower(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.INT64, Shape.of(3, 3), Shape.of(5)));
        var image = lower(context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(Shape.of(1, 1, 3, 3), window(false))),
                DataType.FLOAT32, Shape.of(1, 4, 4), Shape.of(1, 1, 3, 3)));
        assertAll(() -> assertEquals(CpuFoldIr.Family.FOLD_AXIS,
                        ((CpuFoldIr) axis.portableKernelIr()).family()),
                () -> assertEquals(2, axis.boundaryValues().size()),
                () -> assertEquals(5, axis.elementCount()),
                () -> assertTrue(axis.scatterGeometry().isEmpty()),
                () -> assertEquals(3, ((CpuFoldLowering.AxisGeometry)
                        axis.foldGeometry().orElseThrow().mapping()).windowSize()),
                () -> assertEquals(2, ((CpuFoldLowering.TwoDimensionalGeometry)
                        image.foldGeometry().orElseThrow().mapping()).outputColumnsHeight()));
    }

    @Test void admitsExactTypeMatrixAndFailsClosedForBoolIntegral2dAndWrongShapes() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            assertDoesNotThrow(() -> lower(context(new Operation(WindowTransformKind.FOLD_AXIS,
                    new FoldAxisAttrs(0, 3, 1)), type, Shape.of(2, 2), Shape.of(3))));
        }
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 3, 1)),
                        DataType.BOOL, Shape.of(2, 2), Shape.of(3)))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.FOLD2D,
                                new Fold2dAttrs(Shape.of(1, 1, 3, 3), window(false))),
                        DataType.INT32, Shape.of(1, 4, 4), Shape.of(1, 1, 3, 3)))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 4, 1)),
                        DataType.FLOAT64, Shape.of(2, 2), Shape.of(4)))));
    }

    @Test void packsPartialOutputRangeWithoutConcreteGeometryEnteringStructuralIdentity() {
        var first = lower(context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5)));
        var second = lower(context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 6, 2)), DataType.FLOAT32, Shape.of(2, 3), Shape.of(6)));
        long[] packed = first.foldGeometry().orElseThrow().pack(new long[]{4, 7}, 2, 4);
        assertAll(() -> assertEquals(first.portableKernelIr().structuralKey(),
                        second.portableKernelIr().structuralKey()),
                () -> assertEquals(2, packed[4]), () -> assertEquals(4, packed[5]),
                () -> assertEquals(2, packed[10]));
    }

    private static CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionLowering().lower(context);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            DataType type, Shape input, Shape output) {
        return CpuScatterLoweringTest.context(operation, List.of(0),
                List.of(CpuScatterLoweringTest.desc(type, input)),
                CpuScatterLoweringTest.desc(type, output));
    }

    public static Window2dAttrs window(boolean ceil) {
        return new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, ceil);
    }
}
