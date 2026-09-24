package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalAffineCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void admitsExactlyTheFiveTypedAffineMappingsWithCompilerViewLayouts() {
        TensorDescriptor twoThree = canonical(Shape.of(2, 3), false);
        assertTrue(supports(
                new Operation(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(3, 2))),
                twoThree,
                view(Shape.of(3, 2), new long[] {2, 1}, false)));

        TensorDescriptor oneThree = canonical(Shape.of(1, 3), true);
        assertTrue(supports(
                new Operation(ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(2, 3))),
                oneThree,
                view(Shape.of(2, 3), new long[] {0, 1}, true)));

        assertTrue(supports(
                new Operation(AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                twoThree,
                view(Shape.of(3, 2), new long[] {1, 3}, false)));
        assertTrue(supports(
                new Operation(AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(1)),
                twoThree,
                view(Shape.of(2, 1, 3), new long[] {3, 3, 1}, false)));
        assertTrue(supports(
                new Operation(AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(1)),
                canonical(Shape.of(2, 1, 3), false),
                view(Shape.of(2, 3), new long[] {3, 1}, false)));

        Shape rank16 = Shape.of(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3);
        Shape reversed = Shape.of(3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2);
        long[] reversedStrides = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        assertTrue(supports(
                new Operation(AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(
                                15, 14, 13, 12, 11, 10, 9, 8,
                                7, 6, 5, 4, 3, 2, 1, 0))),
                canonical(rank16, false),
                view(reversed, reversedStrides, false)));
    }

    @Test
    void rejectsNoncanonicalInputsAndEveryOutputDescriptorMismatch() {
        TensorDescriptor input = canonical(Shape.of(2, 3), false);
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(Shape.of(3, 2)));
        TensorDescriptor exact = view(Shape.of(3, 2), new long[] {2, 1}, false);

        assertFalse(supports(reshape,
                view(Shape.of(2, 3), new long[] {3, 1}, false), exact));
        assertFalse(supports(reshape, input, canonical(Shape.of(3, 2), false)));
        assertFalse(supports(reshape, input,
                view(Shape.of(3, 2), new long[] {1, 3}, false)));
        assertFalse(supports(reshape, input,
                new TensorDescriptor(DataType.FLOAT64, Shape.of(3, 2),
                        Optional.of(LayoutDescriptor.of(
                                Shape.of(3, 2), new long[] {2, 1}, 0, true)), false)));
        assertFalse(supports(reshape, input,
                view(Shape.of(3, 2), new long[] {2, 1}, true)));
        assertFalse(supports(
                new Operation(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(4, 2))),
                input,
                view(Shape.of(4, 2), new long[] {2, 1}, false)));

        TensorDescriptor expanded = view(Shape.of(2, 3), new long[] {0, 1}, false);
        assertFalse(supports(
                new Operation(AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                expanded,
                view(Shape.of(3, 2), new long[] {1, 0}, false)),
                "affine results must remain terminal leaves");
    }

    @Test
    void rejectsRankZeroZeroExtentAndRankSeventeenBeforePreparation() {
        assertFalse(supports(
                new Operation(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of())),
                canonical(Shape.of(1), false),
                view(Shape.of(), new long[0], false)));
        assertFalse(supports(
                new Operation(ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(2, 0))),
                canonical(Shape.of(1, 0), false),
                view(Shape.of(2, 0), new long[] {0, 1}, false)));
        Shape rank17 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assertFalse(supports(
                new Operation(AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(
                                0, 1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16))),
                canonical(rank17, false),
                view(rank17, LayoutDescriptor.contiguous(rank17).strides(), false)));
    }

    private boolean supports(
            Operation operation, TensorDescriptor input, TensorDescriptor output) {
        return provider.supports(new OperationCapabilityQuery(
                operation, List.of(input), List.of(output)));
    }

    private static TensorDescriptor canonical(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor view(
            Shape shape, long[] strides, boolean requiresGrad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)), requiresGrad);
    }
}
