package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuSpatialWindowLoweringTest {
    private static final Window3dAttrs WINDOW = new Window3dAttrs(
            2, 2, 3, 2, 2, 2, 1, 0, 1, 2, 1, 1, true);

    @Test void namedBatchRecipesResolveTheSameGeometryWithoutWideningPublicAdmission() {
        var b = new DynamicDimension("B");
        Shape image = Shape.ofDimensions(b, new StaticDimension(2), new StaticDimension(3),
                new StaticDimension(4), new StaticDimension(5));
        Shape columns = Shape.ofDimensions(b, new StaticDimension(24), new StaticDimension(12));
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var unfold = CpuNonAffineMovementLoweringTest.context(
                    new Operation(WindowTransformKind.UNFOLD3D, WINDOW), List.of(0),
                    List.of(symbolic(type, image)), symbolic(type, columns));
            var fold = CpuNonAffineMovementLoweringTest.context(
                    new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(image, WINDOW)), List.of(0),
                    List.of(symbolic(type, columns)), symbolic(type, image));
            assertFalse(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    unfold.nodes().getFirst().operation(), List.of(symbolic(type, image)), List.of(symbolic(type, columns)))));
            assertFalse(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    fold.nodes().getFirst().operation(), List.of(symbolic(type, columns)), List.of(symbolic(type, image)))));
            var extraction = new CpuNonAffineMovementLowering().lowerBatch(unfold).batchGeometry().orElseThrow();
            var accumulation = new CpuFoldLowering().lowerBatch(fold).batchGeometry().orElseThrow();
            for (long batch : new long[]{0, 1, 3}) {
                Shape actualImage = Shape.of(batch, 2, 3, 4, 5), actualColumns = Shape.of(batch, 24, 12);
                var staticUnfold = new CpuNonAffineMovementLowering().lower(CpuFoldLoweringTest.context(
                        new Operation(WindowTransformKind.UNFOLD3D, WINDOW), type, actualImage, actualColumns));
                var staticFold = new CpuFoldLowering().lower(CpuFoldLoweringTest.context(
                        new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(actualImage, WINDOW)),
                        type, actualColumns, actualImage));
                var boundUnfold = extraction.bind(batch); var boundFold = accumulation.bind(batch);
                assertEquals(staticUnfold.portableKernelIr().structuralKey(), boundUnfold.portableKernelIr().structuralKey());
                assertEquals(staticFold.portableKernelIr().structuralKey(), boundFold.portableKernelIr().structuralKey());
                assertArrayEquals(staticUnfold.movementGeometry().orElseThrow().pack(new long[]{3, 7}, 0, batch * 288),
                        boundUnfold.movementGeometry().orElseThrow().pack(new long[]{3, 7}, 0, batch * 288));
                assertArrayEquals(staticFold.foldGeometry().orElseThrow().pack(new long[]{3, 7}, 0, batch * 120),
                        boundFold.foldGeometry().orElseThrow().pack(new long[]{3, 7}, 0, batch * 120));
                assertTrue(boundUnfold.movementGeometry().isPresent());
                assertTrue(boundFold.foldGeometry().isPresent());
            }
        }
    }

    @Test void malformedGeometryFailsBeforeGeneration() {
        Shape image = Shape.of(1, 2, 3, 4, 5), columns = Shape.of(1, 24, 12);
        var provider = new CpuCapabilityProvider();
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(WindowTransformKind.UNFOLD3D,
                        new Unfold3dAttrs(WINDOW, ScalarValue.float64(0))),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, image)),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, columns)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(image, WINDOW)),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(1, 24, 11))),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, image)))));
        assertThrows(ArithmeticException.class, () -> CpuSpatialWindowGeometry.of(
                image.toLongArray(), new Window3dAttrs(Long.MAX_VALUE, 1, 1,
                        1, 1, 1, 0, 0, 0, 2, 1, 1, false)));
        assertThrows(IllegalArgumentException.class, () -> CpuSpatialWindowGeometry.of(
                new long[]{1, 1, 1, 1, 1}, WINDOW));
    }

    @Test void newFamiliesCannotReuseTwoDimensionalArtifacts() {
        var unfold3 = new CpuNonAffineMovementLowering().lower(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.UNFOLD3D, WINDOW), DataType.FLOAT32,
                Shape.of(1, 2, 3, 4, 5), Shape.of(1, 24, 12)));
        var two = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        var unfold2 = new CpuNonAffineMovementLowering().lower(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.UNFOLD2D, two), DataType.FLOAT32,
                Shape.of(1, 1, 3, 3), Shape.of(1, 4, 4)));
        assertInstanceOf(CpuDataMovementIr.Unfold3dPlan.class, ((CpuDataMovementIr) unfold3.portableKernelIr()).plan());
        assertNotEquals(unfold2.portableKernelIr().structuralKey(), unfold3.portableKernelIr().structuralKey());
        var f3 = new CpuFoldLowering().lower(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(Shape.of(1, 2, 3, 4, 5), WINDOW)),
                DataType.FLOAT32, Shape.of(1, 24, 12), Shape.of(1, 2, 3, 4, 5)));
        var f2 = new CpuFoldLowering().lower(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.FOLD2D, new Fold2dAttrs(Shape.of(1, 1, 3, 3), two)),
                DataType.FLOAT32, Shape.of(1, 4, 4), Shape.of(1, 1, 3, 3)));
        assertEquals(CpuFoldIr.Family.FOLD3D, ((CpuFoldIr) f3.portableKernelIr()).family());
        assertNotEquals(f2.portableKernelIr().structuralKey(), f3.portableKernelIr().structuralKey());
    }

    private static TensorDescriptor symbolic(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.empty(), false);
    }
}
