package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ScanBodyLogicalLayoutClosureTest {
    @Test
    void closesComputedFloatingAndDiscreteResultsWithoutChangingOrdinaryInference() {
        Operation add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64,
                DataType.BFLOAT16, DataType.FLOAT16, DataType.INT32, DataType.INT64)) {
            boolean grad = type == DataType.FLOAT32 || type == DataType.FLOAT64
                    || type == DataType.BFLOAT16 || type == DataType.FLOAT16;
            TensorDescriptor input = unresolved(type, Shape.of(2, 3), grad);
            List<TensorDescriptor> inputs = List.of(input, input);
            TensorDescriptor ordinary = ScanBody.describeOrdinary(add, inputs, 1).getFirst();
            assertTrue(ordinary.layout().isEmpty());
            assertTrue(ScanBody.hasCanonicalContiguousResults(add));
            TensorDescriptor closed = ScanBody.closeOrdinaryLayouts(add, inputs, List.of(ordinary)).getFirst();
            assertCanonical(closed);
            assertMetadataUnchanged(ordinary, closed);
            assertTrue(ordinary.layout().isEmpty());
            assertTrue(input.layout().isEmpty());
        }

        TensorDescriptor floating = unresolved(DataType.FLOAT64, Shape.of(2, 3), true);
        Operation relu = new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE);
        TensorDescriptor activation = closeInferred(relu, List.of(floating));
        assertCanonical(activation);
        assertSame(DataType.FLOAT64, activation.dataType());
        assertTrue(activation.requiresGrad());

        Operation equal = new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE);
        TensorDescriptor comparison = closeInferred(equal, List.of(floating, floating));
        assertCanonical(comparison);
        assertSame(DataType.BOOL, comparison.dataType());
        assertFalse(comparison.requiresGrad());
    }

    @Test
    void closesEveryCanonicalSiblingInOrderAndPreservesCallerListsAndResolvedDescriptors() {
        Operation topK = new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, true));
        TensorDescriptor input = resolved(DataType.FLOAT32, Shape.of(2, 5), true, 7, 9, 1);
        List<TensorDescriptor> inputs = new ArrayList<>(List.of(input));
        List<TensorDescriptor> inferred = ScanBody.describeOrdinary(topK, inputs, 2);
        List<TensorDescriptor> outputs = new ArrayList<>(inferred);
        List<TensorDescriptor> closed = ScanBody.closeOrdinaryLayouts(topK, inputs, outputs);
        assertEquals(2, closed.size());
        assertCanonical(closed.get(0));
        assertCanonical(closed.get(1));
        assertSame(DataType.FLOAT32, closed.get(0).dataType());
        assertTrue(closed.get(0).requiresGrad());
        assertSame(DataType.INT64, closed.get(1).dataType());
        assertFalse(closed.get(1).requiresGrad());
        for (int slot = 0; slot < inferred.size(); slot++) {
            assertMetadataUnchanged(inferred.get(slot), closed.get(slot));
            assertSame(inferred.get(slot), outputs.get(slot));
            assertTrue(outputs.get(slot).layout().isEmpty());
        }
        assertSame(input, inputs.getFirst());
        assertEquals(1, inputs.size());
        assertArrayEquals(new long[] {9, 1}, input.layout().orElseThrow().strides());
        assertEquals(7, input.layout().orElseThrow().storageOffset());

        TensorDescriptor exact = new TensorDescriptor(inferred.getFirst().dataType(),
                inferred.getFirst().shape(), Optional.of(LayoutDescriptor.of(
                        inferred.getFirst().shape(), new long[] {11, 2}, 19, true)),
                inferred.getFirst().requiresGrad());
        outputs.set(0, exact);
        List<TensorDescriptor> partlyClosed = ScanBody.closeOrdinaryLayouts(topK, inputs, outputs);
        assertSame(exact, partlyClosed.getFirst());
        assertSame(exact.layout().orElseThrow(), partlyClosed.getFirst().layout().orElseThrow());
        assertMetadataUnchanged(exact, partlyClosed.getFirst());
        assertCanonical(partlyClosed.get(1));
        assertSame(exact, outputs.getFirst());
        assertSame(inferred.get(1), outputs.get(1));
    }

    @Test
    void namedAddIsGenericallyMaterializedButHasNoNumericLayoutUntilBound() {
        DynamicDimension batch = new DynamicDimension("B");
        Shape shape = Shape.ofDimensions(batch, new StaticDimension(3));
        TensorDescriptor input = unresolved(DataType.FLOAT32, shape, true);
        Operation add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        List<TensorDescriptor> inputs = List.of(input, input);
        List<TensorDescriptor> outputs = ScanBody.describeOrdinary(add, inputs, 1, Optional.of(batch));
        assertTrue(ScanBody.hasCanonicalContiguousResults(add));
        TensorDescriptor result = ScanBody.closeOrdinaryLayouts(add, inputs, outputs).getFirst();
        assertSame(outputs.getFirst(), result);
        assertSame(batch, result.shape().dimension(0));
        assertTrue(result.layout().isEmpty());
    }

    @Test
    void unknownKindCannotGainLayoutByBorrowingAKnownNameOrSignature() {
        OperationKind unknown = new OperationKind() {
            public String name() { return "RELU"; }
            public List<OperationSignature> signatures() {
                return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
            }
        };
        Operation operation = new Operation(unknown, NoOperationAttrs.INSTANCE);
        TensorDescriptor input = resolved(DataType.FLOAT32, Shape.of(2, 3), true, 5, 3, 1);
        TensorDescriptor output = unresolved(DataType.FLOAT32, input.shape(), true);
        assertFalse(ScanBody.hasCanonicalContiguousResults(operation));
        TensorDescriptor result = ScanBody.closeOrdinaryLayouts(
                operation, List.of(input), List.of(output)).getFirst();
        assertSame(output, result);
        assertTrue(result.layout().isEmpty());
    }

    @Test
    void staticConvolutionsCloseWithoutEnteringTheGenericMaterializationClassifier() {
        Operation conv2d = new Operation(Conv2dKind.CONV2D, Conv2dAttrs.defaults());
        List<TensorDescriptor> inputs2d = List.of(
                unresolved(DataType.FLOAT32, Shape.of(2, 3, 7, 8), true),
                unresolved(DataType.FLOAT32, Shape.of(4, 3, 3, 3), false));
        Operation conv3d = new Operation(Conv3dKind.CONV3D, Conv3dAttrs.defaults());
        List<TensorDescriptor> inputs3d = List.of(
                unresolved(DataType.FLOAT64, Shape.of(2, 3, 6, 7, 8), false),
                unresolved(DataType.FLOAT64, Shape.of(4, 3, 3, 3, 3), true));
        assertFalse(ScanBody.hasCanonicalContiguousResults(conv2d));
        assertFalse(ScanBody.hasCanonicalContiguousResults(conv3d));
        TensorDescriptor result2d = closeInferred(conv2d, inputs2d);
        TensorDescriptor result3d = closeInferred(conv3d, inputs3d);
        assertEquals(Shape.of(2, 4, 5, 6), result2d.shape());
        assertEquals(Shape.of(2, 4, 4, 5, 6), result3d.shape());
        assertCanonical(result2d);
        assertCanonical(result3d);
        assertSame(DataType.FLOAT32, result2d.dataType());
        assertSame(DataType.FLOAT64, result3d.dataType());
        assertTrue(result2d.requiresGrad());
        assertTrue(result3d.requiresGrad());
    }

    @Test
    void permutationAndBroadcastPreserveExactAffineGeometry() {
        TensorDescriptor input = resolved(DataType.FLOAT64, Shape.of(2, 3), true, 5, 10, 2);
        assertClosedView(new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))),
                input, Shape.of(3, 2), 5, 2, 10);
        TensorDescriptor singleton = resolved(DataType.INT64, Shape.of(1, 3), false, 7, 9, 2);
        assertClosedView(new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(Shape.of(4, 3))),
                singleton, Shape.of(4, 3), 7, 0, 2);
    }

    @Test
    void rankEditsPreserveOffsetAndRetainedStrides() {
        TensorDescriptor input = resolved(DataType.FLOAT32, Shape.of(2, 3), true, 5, 10, 2);
        assertClosedView(new Operation(AxisTransformKind.EXPAND_DIMS, new AxisTransformAttrs(1)),
                input, Shape.of(2, 1, 3), 5, 10, 6, 2);
        TensorDescriptor singleton = resolved(DataType.FLOAT32, Shape.of(2, 1, 3), true, 5, 10, 6, 2);
        assertClosedView(new Operation(AxisTransformKind.SQUEEZE, new AxisTransformAttrs(1)),
                singleton, Shape.of(2, 3), 5, 10, 2);
    }

    @Test
    void positiveSliceAndSelectPreserveInheritedStridesAndAdvanceOffset() {
        TensorDescriptor input = resolved(DataType.FLOAT64, Shape.of(4, 6), true, 5, 20, 2);
        Operation slice = new Operation(SliceKind.SLICE,
                new SliceAttrs(List.of(1L), List.of(2L), List.of(1), List.of(2L)));
        assertClosedView(slice, input, Shape.of(4, 2), 7, 20, 4);
        Operation select = new Operation(SelectKind.SELECT, new SelectAttrs(0, 2));
        assertClosedView(select, input, Shape.of(6), 45, 2);
    }

    @Test
    void targetRelativeCropClosesOnlyAtTheExplicitLayoutBoundary() {
        TensorDescriptor input = resolved(DataType.FLOAT64, Shape.of(5, 8), true, 7, 30, 3);
        Shape target = Shape.of(2, 3);
        Shape prefix = Shape.of(1, 2);
        Operation crop = new Operation(SliceKind.SLICE, new CropToShapeAttrs(target, prefix));
        List<TensorDescriptor> inputs = List.of(input);
        Tensor ordinary = TensorFactory.create(input).cropToShape(target, prefix);
        List<TensorDescriptor> described3 = ScanBody.describeOrdinary(crop, inputs, 1);
        List<TensorDescriptor> described4 = ScanBody.describeOrdinary(crop, inputs, 1, Optional.empty());
        assertTrue(ordinary.descriptor().layout().isEmpty());
        assertTrue(described3.getFirst().layout().isEmpty());
        assertTrue(described4.getFirst().layout().isEmpty());
        assertEquals(ordinary.descriptor(), described3.getFirst());
        assertEquals(ordinary.descriptor(), described4.getFirst());
        TensorDescriptor closed = ScanBody.closeOrdinaryLayouts(crop, inputs, described3).getFirst();
        assertMetadataUnchanged(described3.getFirst(), closed);
        assertSame(target, closed.shape());
        assertGeometry(closed, target, 43, true, 30, 3);
        assertTrue(ordinary.descriptor().layout().isEmpty());
        assertTrue(described3.getFirst().layout().isEmpty());
        assertTrue(described4.getFirst().layout().isEmpty());
    }

    @Test
    void signedAndEmptySlicesMaterializeOnlyAtTheExplicitLayoutBoundary() {
        for (var type : DataType.values()) {
            TensorDescriptor input = resolved(type, Shape.of(4, 6), false, 5, 20, 2);
            for (var attrs : List.of(
                    new SliceAttrs(List.of(3L), List.of(4L), List.of(0), List.of(-1L)),
                    new SliceAttrs(List.of(0L), List.of(0L), List.of(0), List.of(1L)),
                    new SliceAttrs(List.of(0L), List.of(0L), List.of(0), List.of(Long.MIN_VALUE)),
                    new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(Long.MIN_VALUE)))) {
                var operation = new Operation(SliceKind.SLICE, attrs);
                var ordinary = ScanBody.describeOrdinary(operation, List.of(input), 1);
                assertTrue(ordinary.getFirst().layout().isEmpty());
                var closed = ScanBody.closeOrdinaryLayouts(operation, List.of(input), ordinary).getFirst();
                assertCanonical(closed);
                assertMetadataUnchanged(ordinary.getFirst(), closed);
                assertTrue(ordinary.getFirst().layout().isEmpty());
                assertSame(closed, ScanBody.closeOrdinaryLayouts(operation, List.of(input),
                        List.of(closed)).getFirst());
            }
        }
    }

    @Test
    void unprovedExtractionsStayUnresolved() {
        TensorDescriptor input = resolved(DataType.FLOAT32, Shape.of(4, 6), true, 5, 20, 2);
        assertUnresolved(new Operation(SliceKind.SLICE,
                new CropToShapeAttrs(Shape.of(0, 2), Shape.of(1, 1))), input);
        assertUnresolved(new Operation(SliceKind.SLICE, new CropToShapeAttrs(Shape.of(2, 2),
                Shape.ofDimensions(new DynamicDimension("P"), new StaticDimension(1)))), input);
        TensorDescriptor noGeometry = unresolved(DataType.FLOAT32, input.shape(), true);
        assertUnresolved(new Operation(SliceKind.SLICE,
                new SliceAttrs(List.of(1L), List.of(2L), List.of(1), List.of(1L))), noGeometry);
        assertUnresolved(new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)), noGeometry);
        TensorDescriptor named = unresolved(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N")), false);
        assertUnresolved(new Operation(SliceKind.SLICE,
                new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(1L))), named);
        assertUnresolved(new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)), named);
    }

    @Test
    void onlyNoncontiguousStaticReshapeMaterializesAndExactContiguousViewRetainsOffset() {
        Operation reshape = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(Shape.of(6)));
        TensorDescriptor strided = resolved(DataType.FLOAT64, Shape.of(2, 3), true, 5, 1, 2);
        List<TensorDescriptor> raw = ScanBody.describeOrdinary(reshape, List.of(strided), 1);
        assertTrue(raw.getFirst().layout().isEmpty());
        TensorDescriptor materialized = ScanBody.closeOrdinaryLayouts(reshape, List.of(strided), raw).getFirst();
        assertMetadataUnchanged(raw.getFirst(), materialized);
        assertCanonical(materialized);
        assertTrue(raw.getFirst().layout().isEmpty());
        TensorDescriptor dense = resolved(DataType.FLOAT64, Shape.of(2, 3), true, 13, 3, 1);
        assertClosedView(reshape, dense, Shape.of(6), 13, 1);
        assertUnresolved(reshape, unresolved(DataType.FLOAT64, Shape.of(2, 3), true));
    }

    private static TensorDescriptor closeInferred(Operation operation, List<TensorDescriptor> inputs) {
        List<TensorDescriptor> outputs = ScanBody.describeOrdinary(operation, inputs, 1);
        TensorDescriptor closed = ScanBody.closeOrdinaryLayouts(operation, inputs, outputs).getFirst();
        assertMetadataUnchanged(outputs.getFirst(), closed);
        return closed;
    }

    private static void assertClosedView(Operation operation, TensorDescriptor input,
            Shape shape, long offset, long... strides) {
        List<TensorDescriptor> inputs = List.of(input);
        TensorDescriptor inferred = ScanBody.describeOrdinary(operation, inputs, 1).getFirst();
        assertGeometry(inferred, shape, offset, true, strides);
        TensorDescriptor captured = unresolved(inferred.dataType(), inferred.shape(), inferred.requiresGrad());
        TensorDescriptor closed = ScanBody.closeOrdinaryLayouts(operation, inputs, List.of(captured)).getFirst();
        assertMetadataUnchanged(captured, closed);
        assertGeometry(closed, shape, offset, true, strides);
        assertEquals(inferred.layout(), closed.layout());
        assertSame(inferred,
                ScanBody.closeOrdinaryLayouts(operation, inputs, List.of(inferred)).getFirst());
        assertTrue(captured.layout().isEmpty());
    }

    private static void assertUnresolved(Operation operation, TensorDescriptor input) {
        List<TensorDescriptor> outputs = ScanBody.describeOrdinary(operation, List.of(input), 1);
        assertTrue(outputs.getFirst().layout().isEmpty());
        TensorDescriptor closed = ScanBody.closeOrdinaryLayouts(operation, List.of(input), outputs).getFirst();
        assertSame(outputs.getFirst(), closed);
        assertTrue(closed.layout().isEmpty());
    }

    private static TensorDescriptor unresolved(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.empty(), grad);
    }

    private static TensorDescriptor resolved(DataType type, Shape shape, boolean grad,
            long offset, long... strides) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), grad);
    }

    private static void assertMetadataUnchanged(TensorDescriptor before, TensorDescriptor after) {
        assertSame(before.dataType(), after.dataType());
        assertSame(before.shape(), after.shape());
        assertEquals(before.requiresGrad(), after.requiresGrad());
    }

    private static void assertCanonical(TensorDescriptor descriptor) {
        assertEquals(LayoutDescriptor.contiguous(descriptor.shape()), descriptor.layout().orElseThrow());
        assertFalse(descriptor.layout().orElseThrow().isView());
    }

    private static void assertGeometry(TensorDescriptor descriptor, Shape shape,
            long offset, boolean view, long... strides) {
        assertEquals(shape, descriptor.shape());
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        assertEquals(offset, layout.storageOffset());
        assertArrayEquals(strides, layout.strides());
        assertEquals(view, layout.isView());
    }
}
