package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.shape.Dimension;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;

import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;




import java.util.Optional;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import org.junit.jupiter.api.Test;

class TensorSlicePlacementExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(40_000_000);

    

    @Test
    void sliceUpdateAcceptsEveryTypeAndValidGradientCombination() {
        for (DataType dataType : DataType.values()) {
            for (boolean baseGrad : validGradientChoices(dataType)) {
                for (boolean updateGrad : validGradientChoices(dataType)) {
                    Shape baseShape = Shape.of(3, 6);
                    Shape updateShape = Shape.of(3, 2);
                    Tensor base = tensor(dataType, baseShape, trueLayout(baseShape), baseGrad);
                    Tensor update = tensor(dataType, updateShape, Optional.empty(), updateGrad);

                    Tensor result = base.sliceUpdate(
                            update, new long[] {1}, new int[] {1}, new long[] {2});
                    Shape prefixShape = Shape.of(0, 1);
                    Tensor targetRelative = base.sliceUpdate(update, prefixShape);

                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertSame(baseShape, result.descriptor().shape()),
                            () -> assertEquals(
                                    baseGrad || updateGrad,
                                    result.descriptor().requiresGrad()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()),
                            () -> assertSame(dataType, targetRelative.descriptor().dataType()),
                            () -> assertSame(baseShape, targetRelative.descriptor().shape()),
                            () -> assertEquals(
                                    baseGrad || updateGrad,
                                    targetRelative.descriptor().requiresGrad()),
                            () -> assertTrue(targetRelative.descriptor().layout().isEmpty()),
                            () -> assertTrue(targetRelative.label().isEmpty()),
                            () -> assertTrue(targetRelative.hostStorage().isEmpty()));
                }
            }
        }
    }

    @Test
    void sliceUpdateNormalizesSignedMultiAxisCoordinatesAndSnapshotsArrays() {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(5, 6), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false);
        long[] starts = {-1, 1};
        int[] axes = {-2, -1};
        long[] steps = {-2, 2};

        Tensor result = base.sliceUpdate(update, starts, axes, steps);
        SliceAttrs attrs = (SliceAttrs) result.provenance().orElseThrow().operation().attrs();
        starts[0] = 0;
        axes[0] = 1;
        steps[0] = 1;

        assertAll(
                () -> assertEquals(List.of(4L, 1L), attrs.starts()),
                () -> assertEquals(List.of(3L, 2L), attrs.lengths()),
                () -> assertEquals(List.of(0, 1), attrs.axes()),
                () -> assertEquals(List.of(-2L, 2L), attrs.steps()),
                () -> assertArrayEquals(new long[] {0, 1}, starts),
                () -> assertArrayEquals(new int[] {1, -1}, axes),
                () -> assertArrayEquals(new long[] {1, 2}, steps));
    }

    @Test
    void sliceUpdateRecordsExactOperationProducerAndProvenance() {
        Shape baseShape = Shape.of(3, 6);
        Tensor base = tensor(DataType.FLOAT64, baseShape, Optional.empty(), false);
        Tensor update = tensor(DataType.FLOAT64, Shape.of(3, 2), Optional.empty(), true);

        Tensor result = base.sliceUpdate(
                update, new long[] {1}, new int[] {1}, new long[] {2});
        TensorProvenance provenance = result.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();

        assertAll(
                () -> assertSame(SliceKind.SLICE_UPDATE, provenance.operation().kind()),
                () -> assertEquals(
                        OperationSignature.fixed(SliceAttrs.class, 2, 1),
                        provenance.operation().signature()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(base, producer.inputs().get(0)),
                () -> assertSame(update, producer.inputs().get(1)),
                () -> assertEquals(1, producer.outputCount()),
                () -> assertSame(result.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void targetRelativeSliceUpdateRetainsExactShapesOperationProducerAndProvenance() {
        Shape baseShape = Shape.of(5, 8);
        Shape updateShape = Shape.of(2, 3);
        Shape prefixShape = Shape.of(1, 4);
        Tensor base = tensor(
                DataType.FLOAT64,
                baseShape,
                Optional.of(LayoutDescriptor.contiguous(baseShape)),
                false);
        Tensor update = tensor(DataType.FLOAT64, updateShape, Optional.empty(), true);

        Tensor result = base.sliceUpdate(update, prefixShape);
        TensorProvenance provenance = result.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();
        CropToShapeAttrs attrs = (CropToShapeAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(baseShape, result.descriptor().shape()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertSame(SliceKind.SLICE_UPDATE, provenance.operation().kind()),
                () -> assertEquals(
                        OperationSignature.fixed(CropToShapeAttrs.class, 2, 1),
                        provenance.operation().signature()),
                () -> assertSame(updateShape, attrs.targetShape()),
                () -> assertSame(prefixShape, attrs.prefixShape()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(base, producer.inputs().get(0)),
                () -> assertSame(update, producer.inputs().get(1)),
                () -> assertEquals(1, producer.outputCount()),
                () -> assertSame(result, producer.output(0)),
                () -> assertSame(result.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertNotSame(base, result),
                () -> assertNotSame(update, result));
    }

    @Test
    void targetRelativeSliceUpdateAcceptsScalarZeroAndDefersEveryUnresolvedFitCategory() {
        Tensor scalarBase = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor scalarUpdate = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor zeroBase = tensor(DataType.INT32, Shape.of(3), Optional.empty(), false);
        Tensor zeroUpdate = tensor(DataType.INT32, Shape.of(0), Optional.empty(), false);
        Dimension baseN = new DynamicDimension("B");
        Dimension prefixN = new DynamicDimension("P");
        Dimension updateN = DimensionExpressions.addConstant(new DynamicDimension("U"), 1);
        Tensor unresolvedBase = tensor(
                DataType.FLOAT32, Shape.ofDimensions(baseN), Optional.empty(), true);
        Tensor staticBase = tensor(DataType.FLOAT32, Shape.of(4), Optional.empty(), true);
        Tensor unresolvedUpdate = tensor(
                DataType.FLOAT32, Shape.ofDimensions(updateN), Optional.empty(), false);
        Tensor staticUpdate = tensor(DataType.FLOAT32, Shape.of(9), Optional.empty(), false);

        Tensor scalar = scalarBase.sliceUpdate(scalarUpdate, Shape.scalar());
        Tensor zero = zeroBase.sliceUpdate(zeroUpdate, Shape.of(3));
        Tensor dynamicBase = unresolvedBase.sliceUpdate(staticUpdate, Shape.of(8));
        Shape dynamicPrefixShape = Shape.ofDimensions(prefixN);
        Tensor dynamicPrefix = staticBase.sliceUpdate(staticUpdate, dynamicPrefixShape);
        Tensor dynamicUpdate = staticBase.sliceUpdate(unresolvedUpdate, Shape.of(8));

        assertAll(
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertSame(zeroBase.descriptor().shape(), zero.descriptor().shape()),
                () -> assertSame(
                        unresolvedBase.descriptor().shape(), dynamicBase.descriptor().shape()),
                () -> assertSame(
                        staticBase.descriptor().shape(), dynamicPrefix.descriptor().shape()),
                () -> assertSame(
                        staticBase.descriptor().shape(), dynamicUpdate.descriptor().shape()),
                () -> assertSame(
                        dynamicPrefixShape,
                        ((CropToShapeAttrs) dynamicPrefix.provenance().orElseThrow()
                                .operation().attrs()).prefixShape()),
                () -> assertSame(
                        unresolvedUpdate.descriptor().shape(),
                        ((CropToShapeAttrs) dynamicUpdate.provenance().orElseThrow()
                                .operation().attrs()).targetShape()));
    }

    @Test
    void targetRelativeSliceUpdateValidatesExactOrderMessagesAndNoIds() throws Exception {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true);
        Tensor wrongRank = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(null, null, (Shape) null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(base, null, (Shape) null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(base, update, (Shape) null));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(wrongType, Shape.of(0, 0)));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(wrongRank, Shape.of(0)));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, Shape.of(0)));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, Shape.of(2, 4)));

        assertEquals(before, next.get());
    }

    @Test
    void targetRelativeSliceUpdatePropagatesStaticFitOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor base = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor update = tensor(DataType.INT64, Shape.of(1), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> base.sliceUpdate(update, Shape.of(Long.MAX_VALUE)));

        assertEquals(before, next.get());
    }

    @Test
    void sliceUpdateCreatesExplicitFullReplacementAndCanonicalEmptyRegions() {
        Tensor base = tensor(DataType.INT64, Shape.of(3, 6), Optional.empty(), false);
        Tensor full = tensor(DataType.INT64, Shape.of(3, 6), Optional.empty(), false);
        Tensor empty = tensor(DataType.INT64, Shape.of(3, 0), Optional.empty(), false);

        Tensor fullResult = base.sliceUpdate(
                full, new long[0], new int[0], new long[0]);
        Tensor emptyResult = base.sliceUpdate(
                empty,
                new long[] {Long.MIN_VALUE},
                new int[] {1},
                new long[] {Long.MIN_VALUE});
        SliceAttrs fullAttrs = (SliceAttrs) fullResult.provenance().orElseThrow().operation().attrs();
        SliceAttrs emptyAttrs =
                (SliceAttrs) emptyResult.provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(List.of(), fullAttrs.starts()),
                () -> assertSame(base.descriptor().shape(), fullResult.descriptor().shape()),
                () -> assertNotSame(base, fullResult),
                () -> assertEquals(List.of(0L), emptyAttrs.starts()),
                () -> assertEquals(List.of(0L), emptyAttrs.lengths()),
                () -> assertEquals(List.of(Long.MIN_VALUE), emptyAttrs.steps()),
                () -> assertNotSame(base, emptyResult));
    }

    @Test
    void sliceUpdateDefersOnlyDynamicUpperBoundsAndRequiresStaticSelectedUpdateExtents() {
        Dimension dynamic = new DynamicDimension("N");
        Shape baseShape = Shape.ofDimensions(dynamic, new StaticDimension(4));
        Tensor base = tensor(DataType.FLOAT32, baseShape, Optional.empty(), true);
        Tensor valid = tensor(DataType.FLOAT32, Shape.of(2, 4), Optional.empty(), false);
        Tensor dynamicSelected = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("U"), new StaticDimension(4)),
                Optional.empty(),
                false);

        Tensor deferred = base.sliceUpdate(
                valid, new long[] {10}, new int[] {0}, new long[] {1});
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        valid, new long[] {-1}, new int[] {0}, new long[] {1}));
        IllegalArgumentException negativeLast = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        valid, new long[] {0}, new int[] {0}, new long[] {-1}));
        IllegalArgumentException selectedDynamic = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        dynamicSelected, new long[] {0}, new int[] {0}, new long[] {1}));

        assertAll(
                        () -> assertSame(baseShape, deferred.descriptor().shape()));
    }

    @Test
    void sliceUpdateRejectsExactTypeRankEntryCoordinateAndShapeFailuresInOrder()
            throws Exception {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(3, 2), Optional.empty(), true);
        Tensor wrongRank = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), true);
        Tensor wrongShape = tensor(DataType.FLOAT32, Shape.of(4, 2), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(null, null, null, null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(base, null, null, null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(base, update, null, null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(base, update, new long[0], null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.update(
                base, update, new long[0], new int[0], null));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, new long[] {0}, new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(wrongType, new long[0], new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(wrongRank, new long[0], new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, new long[] {0}, new int[] {-3}, new long[] {1}));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(
                tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true),
                new long[] {0, 0}, new int[] {0, -2}, new long[] {1, 1}));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, new long[] {0}, new int[] {1}, new long[] {0}));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(update, new long[] {5}, new int[] {1}, new long[] {1}));
        assertFailure(IllegalArgumentException.class, () -> base.sliceUpdate(wrongShape, new long[] {0}, new int[] {1}, new long[] {1}));

        assertEquals(before, next.get());
    }

    @Test
    void sliceUpdatePropagatesCheckedCoordinateOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor base = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor update = tensor(DataType.INT64, Shape.of(3), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> base.sliceUpdate(
                        update,
                        new long[] {Long.MAX_VALUE - 1},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE}));

        assertEquals(before, next.get());
    }

    @Test
    void cropRetainsExactStaticAndSymbolicShapesAndMetadata() {
        Dimension n = new DynamicDimension("N");
        Shape inputShape = Shape.ofDimensions(DimensionExpressions.addConstant(n, 3));
        Shape targetShape = Shape.ofDimensions(n);
        Shape prefixShape = Shape.of(1);
        Tensor input = tensor(DataType.FLOAT64, inputShape, Optional.empty(), true);

        Tensor result = input.cropToShape(targetShape, prefixShape);
        TensorProvenance provenance = result.provenance().orElseThrow();
        CropToShapeAttrs attrs = (CropToShapeAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(targetShape, result.descriptor().shape()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                () -> assertEquals(
                        OperationSignature.fixed(CropToShapeAttrs.class, 1, 1),
                        provenance.operation().signature()),
                () -> assertSame(targetShape, attrs.targetShape()),
                () -> assertSame(prefixShape, attrs.prefixShape()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void cropAcceptsScalarZeroAndAllUnresolvedBoundCategories() {
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor zero = tensor(DataType.INT32, Shape.of(3), Optional.empty(), false);
        Dimension n = new DynamicDimension("N");
        Tensor dynamicInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(n), Optional.empty(), true);
        Tensor staticInput = tensor(DataType.FLOAT32, Shape.of(4), Optional.empty(), true);

        Tensor scalarResult = scalar.cropToShape(Shape.scalar(), Shape.scalar());
        Tensor zeroResult = zero.cropToShape(Shape.of(0), Shape.of(3));
        Tensor dynamicSource = dynamicInput.cropToShape(Shape.of(9), Shape.of(8));
        Tensor dynamicTarget = staticInput.cropToShape(
                Shape.ofDimensions(new DynamicDimension("T")), Shape.of(8));
        Tensor dynamicPrefix = staticInput.cropToShape(
                Shape.of(9), Shape.ofDimensions(new DynamicDimension("P")));

        assertAll(
                () -> assertSame(Shape.scalar(), scalarResult.descriptor().shape()),
                () -> assertEquals(Shape.of(0), zeroResult.descriptor().shape()),
                () -> assertEquals(Shape.of(9), dynamicSource.descriptor().shape()),
                () -> assertEquals("Shape[T]", dynamicTarget.descriptor().shape().toString()),
                () -> assertEquals(Shape.of(9), dynamicPrefix.descriptor().shape()));
    }

    @Test
    void cropValidatesNullRankAndFirstStaticBoundInExactOrderWithoutIds()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.cropToShape(null, null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.cropToShape(input, null, null));
        assertFailure(NullPointerException.class, () -> TensorSlicePlacementExpressions.cropToShape(
                input, Shape.of(1, 1), null));
        assertFailure(IllegalArgumentException.class, () -> input.cropToShape(Shape.of(1), Shape.of(1)));
        assertFailure(IllegalArgumentException.class, () -> input.cropToShape(Shape.of(1, 1), Shape.of(1)));
        assertFailure(IllegalArgumentException.class, () -> input.cropToShape(Shape.of(2, 7), Shape.of(2, 0)));

        assertEquals(before, next.get());
    }

    @Test
    void cropPropagatesCheckedStaticBoundOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor input = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> input.cropToShape(Shape.of(1), Shape.of(Long.MAX_VALUE)));

        assertEquals(before, next.get());
    }

    @Test
    void successfulCallsConsumeOneFreshIdEachAndLeaveInputsUnchanged() throws Exception {
        Shape baseShape = Shape.of(3, 6);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(baseShape);
        Tensor base = tensor(DataType.FLOAT32, baseShape, Optional.of(layout), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false);
        TensorDescriptor baseDescriptor = base.descriptor();
        TensorDescriptor updateDescriptor = update.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor updated = base.sliceUpdate(
                update, new long[] {1}, new int[] {1}, new long[] {2});
        Tensor targetRelative = base.sliceUpdate(update, Shape.of(0, 1));
        Tensor cropped = base.cropToShape(Shape.of(2, 3), Shape.of(1, 2));

        assertAll(
                () -> assertEquals(before, updated.id().value()),
                () -> assertEquals(before + 1, targetRelative.id().value()),
                () -> assertEquals(before + 2, cropped.id().value()),
                () -> assertEquals(before + 3, next.get()),
                () -> assertSame(baseDescriptor, base.descriptor()),
                () -> assertSame(updateDescriptor, update.descriptor()),
                () -> assertSame(layout, base.descriptor().layout().orElseThrow()),
                () -> assertTrue(base.provenance().isEmpty()),
                () -> assertTrue(update.provenance().isEmpty()),
                () -> assertNotEquals(updated.id(), cropped.id()),
                () -> assertNotSame(
                        updated.provenance().orElseThrow().producer(),
                        targetRelative.provenance().orElseThrow().producer()),
                () -> assertNotSame(
                        targetRelative.provenance().orElseThrow().producer(),
                        cropped.provenance().orElseThrow().producer()));
    }

    @Test
    void targetRelativeSliceUpdateExhaustionRemainsFinal() throws Exception {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(1), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> base.sliceUpdate(update, Shape.of(1)));

            assertAll(
                            () -> assertEquals(Long.MAX_VALUE, next.get()),
                            () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    

    private static Optional<LayoutDescriptor> trueLayout(Shape shape) {
        return Optional.of(LayoutDescriptor.contiguous(shape));
    }

    private static List<Boolean> validGradientChoices(DataType dataType) {
        return dataType.isDifferentiable() ? List.of(false, true) : List.of(false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static <T extends Throwable> void assertFailure(
                Class<T> type,
                org.junit.jupiter.api.function.Executable executable) {
            assertThrows(type, executable);
        }
}
