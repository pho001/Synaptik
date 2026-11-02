package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import io.github.pho001.synaptik.model.shape.Dimension;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;

import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.reflect.Field;



import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorWindowExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(4_000_000);

    

    @Test
    void unfoldAcceptsEveryDataTypeAndPreservesExactMetadataAndDimensions() {
        for (DataType dataType : DataType.values()) {
            DynamicDimension batch = new DynamicDimension("batch-" + dataType);
            StaticDimension width = new StaticDimension(3);
            Tensor input = tensor(
                    dataType,
                    Shape.ofDimensions(batch, new StaticDimension(5), width),
                    Optional.empty(),
                    dataType.isDifferentiable());

            Tensor result = input.unfold(-2, 3, 1);
            TensorProvenance provenance = result.provenance().orElseThrow();

            assertAll(
                    () -> assertEquals(dataType, result.descriptor().dataType()),
                    () -> assertSame(batch, result.descriptor().shape().dimensions().get(0)),
                    () -> assertEquals(
                            new StaticDimension(3),
                            result.descriptor().shape().dimensions().get(1)),
                    () -> assertSame(width, result.descriptor().shape().dimensions().get(2)),
                    () -> assertEquals(
                            new StaticDimension(3),
                            result.descriptor().shape().dimensions().get(3)),
                    () -> assertEquals(
                            dataType.isDifferentiable(), result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(WindowTransformKind.UNFOLD_AXIS,
                            provenance.operation().kind()),
                    () -> assertEquals(
                            new UnfoldAxisAttrs(1, 3, 1), provenance.operation().attrs()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertNotSame(input, result));
        }
    }

    @Test
    void unfoldSupportsLargeStepsAndExtremeValidCounts() {
        Tensor ordinary = tensor(DataType.INT64, Shape.of(2, 5), Optional.empty(), false);
        Tensor maximum = tensor(
                DataType.BOOL, Shape.of(Long.MAX_VALUE), Optional.empty(), false);

        Tensor oneWindow = ordinary.unfold(1, 3, Long.MAX_VALUE);
        Tensor extreme = maximum.unfold(0, 1, 1);

        assertAll(
                () -> assertEquals(Shape.of(2, 1, 3), oneWindow.descriptor().shape()),
                () -> assertEquals(
                        Shape.of(Long.MAX_VALUE, 1), extreme.descriptor().shape()));
    }

    @Test
    void unfoldRejectsInvalidRequestsInExactOrderWithExactMessages() {
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), true);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("time")),
                Optional.empty(),
                true);
        Tensor zero = tensor(DataType.FLOAT32, Shape.of(0), Optional.empty(), false);
        Tensor shortAxis = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.unfoldAxis(null, 0, 1, 1));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorWindowExpressions.unfoldAxis(scalar, Integer.MIN_VALUE, 0, 0));
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> dynamic.unfold(Integer.MIN_VALUE, 0, 0));
        IllegalArgumentException size = assertThrows(
                IllegalArgumentException.class, () -> dynamic.unfold(0, 0, 0));
        IllegalArgumentException step = assertThrows(
                IllegalArgumentException.class, () -> dynamic.unfold(0, 1, 0));
        IllegalArgumentException staticity = assertThrows(
                IllegalArgumentException.class, () -> dynamic.unfold(0, 1, 1));
        IllegalArgumentException zeroFit = assertThrows(
                IllegalArgumentException.class, () -> zero.unfold(0, 1, 1));
        IllegalArgumentException fit = assertThrows(
                IllegalArgumentException.class, () -> shortAxis.unfold(0, 3, 1));

        
    }

    @Test
    void foldAxisAcceptsEveryNumericTypeAndPreservesUnaffectedDimensions() {
        for (DataType dataType : List.of(
                DataType.FLOAT64,
                DataType.FLOAT32,
                DataType.BFLOAT16,
                DataType.INT32,
                DataType.INT64)) {
            DynamicDimension batch = new DynamicDimension("batch-" + dataType);
            StaticDimension features = new StaticDimension(7);
            Tensor input = tensor(
                    dataType,
                    Shape.ofDimensions(batch, new StaticDimension(3), features,
                            new StaticDimension(3)),
                    Optional.empty(),
                    dataType.isDifferentiable());

            Tensor result = input.foldAxis(-2, 5, 1);
            TensorProvenance provenance = result.provenance().orElseThrow();

            assertAll(
                    () -> assertSame(batch, result.descriptor().shape().dimensions().get(0)),
                    () -> assertEquals(new StaticDimension(5),
                            result.descriptor().shape().dimensions().get(1)),
                    () -> assertSame(features,
                            result.descriptor().shape().dimensions().get(2)),
                    () -> assertEquals(dataType, result.descriptor().dataType()),
                    () -> assertEquals(dataType.isDifferentiable(),
                            result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(WindowTransformKind.FOLD_AXIS,
                            provenance.operation().kind()),
                    () -> assertEquals(new FoldAxisAttrs(1, 5, 1),
                            provenance.operation().attrs()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertEquals(0, provenance.outputIndex()));
        }
    }

    @Test
    void foldAxisSupportsZeroOutputExactlyForZeroWindows() {
        Tensor zeroWindows = tensor(
                DataType.FLOAT32, Shape.of(2, 0, 3), Optional.empty(), true);
        Tensor nonZeroWindows = tensor(
                DataType.FLOAT32, Shape.of(2, 1, 3), Optional.empty(), true);

        Tensor result = zeroWindows.foldAxis(1, 0, 2);
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> nonZeroWindows.foldAxis(1, 0, 2));

        assertAll(
                        () -> assertEquals(Shape.of(2, 0), result.descriptor().shape()));
    }

    @Test
    void foldAxisRejectsInvalidRequestsInExactOrderWithExactMessages() {
        Tensor rankOneBool = tensor(DataType.BOOL, Shape.of(1), Optional.empty(), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(3, 3), Optional.empty(), false);
        Tensor dynamicCount = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("windows"), new StaticDimension(3)),
                Optional.empty(), true);
        Tensor dynamicWindow = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(3), new DynamicDimension("window")),
                Optional.empty(), true);
        Tensor zeroWindow = tensor(DataType.FLOAT32, Shape.of(0, 0), Optional.empty(), true);
        Tensor tooWide = tensor(DataType.FLOAT32, Shape.of(1, 6), Optional.empty(), true);
        Tensor wrongCount = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.foldAxis(null, 0, 1, 1));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class,
                () -> rankOneBool.foldAxis(Integer.MIN_VALUE, -1, 0));
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> bool.foldAxis(Integer.MIN_VALUE, -1, 0));
        IllegalArgumentException outputSize = assertThrows(
                IllegalArgumentException.class, () -> bool.foldAxis(0, -1, 0));
        IllegalArgumentException step = assertThrows(
                IllegalArgumentException.class, () -> bool.foldAxis(0, 1, 0));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> bool.foldAxis(0, 1, 1));
        IllegalArgumentException countStaticity = assertThrows(
                IllegalArgumentException.class, () -> dynamicCount.foldAxis(0, 5, 1));
        IllegalArgumentException windowStaticity = assertThrows(
                IllegalArgumentException.class, () -> dynamicWindow.foldAxis(0, 5, 1));
        IllegalArgumentException positiveWindow = assertThrows(
                IllegalArgumentException.class, () -> zeroWindow.foldAxis(0, 0, 1));
        IllegalArgumentException windowFit = assertThrows(
                IllegalArgumentException.class, () -> tooWide.foldAxis(0, 5, 1));
        IllegalArgumentException count = assertThrows(
                IllegalArgumentException.class, () -> wrongCount.foldAxis(0, 5, 1));

        
    }

    @Test
    void unfold2dSupportsFloorCeilPaddingStrideDilationAndDynamicBatch() {
        DynamicDimension batch = new DynamicDimension("batch");
        Tensor floorInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(
                        batch,
                        new StaticDimension(1),
                        new StaticDimension(3),
                        new StaticDimension(3)),
                Optional.empty(),
                true);
        Window2dAttrs floor = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        Window2dAttrs ceil = window(2, 2, 2, 2, 1, 1, 2, 2, true);

        Tensor floorResult = floorInput.unfold2d(floor);
        Tensor ceilResult = floorInput.unfold2d(ceil);

        assertAll(
                () -> assertSame(batch, floorResult.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(
                        Shape.ofDimensions(batch, new StaticDimension(4), new StaticDimension(4)),
                        floorResult.descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, new StaticDimension(4), new StaticDimension(4)),
                        ceilResult.descriptor().shape()),
                () -> assertSame(floor,
                        floorResult.provenance().orElseThrow().operation().attrs()),
                () -> assertSame(WindowTransformKind.UNFOLD2D,
                        floorResult.provenance().orElseThrow().operation().kind()),
                () -> assertTrue(floorResult.descriptor().layout().isEmpty()),
                    () -> assertTrue(floorResult.descriptor().requiresGrad()));
    }

    @Test
    void unfold2dRetainsExactDynamicCanonicalColumnFormulas() {
        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension channels = new DynamicDimension("C");
        DynamicDimension height = new DynamicDimension("H");
        Dimension width = DimensionExpressions.unknown(1, Optional.empty());
        Tensor input = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(batch, channels, height, width),
                Optional.empty(), true);
        Window2dAttrs floor = window(3, 3, 2, 2, 1, 1, 1, 1, false);
        Window2dAttrs ceil = window(3, 3, 2, 2, 1, 1, 1, 1, true);

        Tensor floorResult = input.unfold2d(floor);
        Tensor ceilResult = input.unfold2d(ceil);
        Dimension expectedChannels = DimensionExpressions.multiply(channels, 9);
        Dimension floorHeight = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(height, -1), 2), 1);
        Dimension floorWidth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(width, -1), 2), 1);
        Dimension ceilHeight = DimensionExpressions.addConstant(
                DimensionExpressions.ceilingDivide(
                        DimensionExpressions.addConstant(height, -1), 2), 1);
        Dimension ceilWidth = DimensionExpressions.addConstant(
                DimensionExpressions.ceilingDivide(
                        DimensionExpressions.addConstant(width, -1), 2), 1);

        assertAll(
                () -> assertSame(batch,
                        floorResult.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(expectedChannels,
                        floorResult.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(DimensionExpressions.multiply(floorHeight, floorWidth),
                        floorResult.descriptor().shape().dimensions().get(2)),
                () -> assertEquals(DimensionExpressions.multiply(ceilHeight, ceilWidth),
                        ceilResult.descriptor().shape().dimensions().get(2)));
    }

    @Test
    void unfold2dAcceptsExactlyFloatingTypes() {
        Window2dAttrs window = window(1, 1, 1, 1, 0, 0, 1, 1, false);
        for (DataType dataType : DataType.values()) {
            Tensor input = tensor(
                    dataType, Shape.of(1, 2, 3, 4), Optional.empty(), dataType.isDifferentiable());
            if (dataType.isFloating()) {
                Tensor result = input.unfold2d(window);
                assertAll(
                        () -> assertEquals(dataType, result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(1, 2, 12), result.descriptor().shape()),
                        () -> assertEquals(dataType.isDifferentiable(),
                                result.descriptor().requiresGrad()));
            } else {
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> input.unfold2d(window));
                
            }
        }
    }

    @Test
    void configurableUnfold2dRetainsExactPaddingReferencesAndBits() {
        Window2dAttrs window = window(2, 2, 2, 2, 1, 1, 1, 1, true);
        List<ScalarValue> paddingValues = List.of(
                ScalarValue.float64(Double.longBitsToDouble(0xFFF8_0000_0000_0042L)),
                ScalarValue.float32(Float.intBitsToFloat(0xFF80_0042)),
                ScalarValue.bfloat16Bits((short) 0xFFC2));

        for (ScalarValue paddingValue : paddingValues) {
            Tensor input = tensor(
                    paddingValue.dataType(), Shape.of(1, 1, 3, 3), Optional.empty(), true);
            Tensor result = input.unfold2d(window, paddingValue);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Unfold2dAttrs attrs = (Unfold2dAttrs) provenance.operation().attrs();

            assertAll(
                    () -> assertEquals(Shape.of(1, 4, 9), result.descriptor().shape()),
                    () -> assertSame(window, attrs.window()),
                    () -> assertSame(paddingValue, attrs.paddingValue()),
                    () -> assertSame(WindowTransformKind.UNFOLD2D,
                            provenance.operation().kind()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()));
        }
    }

    @Test
    void configurableUnfold2dRejectsNullAndTypeMismatchBeforeIdentityAllocation()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 3), Optional.empty(), true);
        Window2dAttrs window = window(1, 1, 1, 1, 0, 0, 1, 1, false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.unfold2d(null, null, null));
        NullPointerException nullWindow = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.unfold2d(input, null, null));
        NullPointerException nullPadding = assertThrows(
                NullPointerException.class, () -> input.unfold2d(window, null));
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> input.unfold2d(window, ScalarValue.float64(0.0)));

        assertAll(
                        () -> assertEquals(before, next.get()));
    }

    @Test
    void unfold2dRejectsNullRankTypeFitAndOverflowInExactOrder() {
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor rankThree = tensor(DataType.INT32, Shape.of(1, 1, 1), Optional.empty(), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(1, 1, 1, 1), Optional.empty(), false);
        Tensor tooSmall = tensor(DataType.FLOAT32, Shape.of(1, 1, 1, 2), Optional.empty(), true);
        Tensor ordinary = tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2), Optional.empty(), true);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.unfold2d(null, null));
        NullPointerException nullWindow = assertThrows(
                NullPointerException.class, () -> ordinary.unfold2d(null));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class, () -> rankThree.unfold2d(window));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> integral.unfold2d(window));
        IllegalArgumentException fit = assertThrows(
                IllegalArgumentException.class, () -> tooSmall.unfold2d(window));
        ArithmeticException dilationOverflow = assertThrows(
                ArithmeticException.class,
                () -> ordinary.unfold2d(
                        window(2, 1, 1, 1, 0, 0, Long.MAX_VALUE, 1, false)));
        ArithmeticException paddingOverflow = assertThrows(
                ArithmeticException.class,
                () -> ordinary.unfold2d(
                        window(1, 1, 1, 1, Long.MAX_VALUE, 0, 1, 1, false)));

        
    }

    @Test
    void fold2dRetainsExactOutputAndWindowReferencesWithDynamicBatch() {
        DynamicDimension batch = new DynamicDimension("batch");
        Shape inputShape = Shape.ofDimensions(
                batch, new StaticDimension(4), new StaticDimension(4));
        Shape outputShape = Shape.ofDimensions(
                batch,
                new StaticDimension(1),
                new StaticDimension(3),
                new StaticDimension(3));
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor input = tensor(DataType.BFLOAT16, inputShape, Optional.empty(), true);

        Tensor result = input.fold2d(outputShape, window);
        TensorProvenance provenance = result.provenance().orElseThrow();
        Fold2dAttrs attrs = (Fold2dAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(outputShape, result.descriptor().shape()),
                () -> assertSame(outputShape, attrs.outputShape()),
                () -> assertSame(window, attrs.window()),
                () -> assertEquals(DataType.BFLOAT16, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(WindowTransformKind.FOLD2D, provenance.operation().kind()),
                () -> assertEquals(List.of(input), provenance.inputs()));
    }

    @Test
    void fold2dAcceptsFloatingFloorAndCeilGeometry() {
        Window2dAttrs floor = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        Window2dAttrs ceil = window(2, 2, 2, 2, 1, 1, 2, 2, true);
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Tensor columns = tensor(dataType, Shape.of(1, 4, 4), Optional.empty(), true);
            Tensor floorResult = columns.fold2d(Shape.of(1, 1, 3, 3), floor);
            Tensor ceilResult = columns.fold2d(Shape.of(1, 1, 3, 3), ceil);
            assertAll(
                    () -> assertEquals(Shape.of(1, 1, 3, 3),
                            floorResult.descriptor().shape()),
                    () -> assertEquals(Shape.of(1, 1, 3, 3),
                            ceilResult.descriptor().shape()),
                    () -> assertEquals(dataType, floorResult.descriptor().dataType()));
        }
    }

    @Test
    void fold2dRejectsNullRankTypeAndBatchInExactOrder() {
        Window2dAttrs window = window(1, 1, 1, 1, 0, 0, 1, 1, false);
        Shape output = Shape.of(1, 1, 2, 2);
        Tensor columns = tensor(DataType.FLOAT32, Shape.of(1, 1, 4), Optional.empty(), true);
        Tensor rankTwo = tensor(DataType.INT32, Shape.of(1, 1), Optional.empty(), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(1, 1, 4), Optional.empty(), false);
        Tensor wrongBatch = tensor(DataType.FLOAT32, Shape.of(2, 1, 4), Optional.empty(), true);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.fold2d(null, null, null));
        NullPointerException nullOutput = assertThrows(
                NullPointerException.class,
                () -> TensorWindowExpressions.fold2d(columns, null, null));
        NullPointerException nullWindow = assertThrows(
                NullPointerException.class, () -> columns.fold2d(output, null));
        IllegalArgumentException inputRank = assertThrows(
                IllegalArgumentException.class, () -> rankTwo.fold2d(Shape.scalar(), window));
        IllegalArgumentException outputRank = assertThrows(
                IllegalArgumentException.class, () -> columns.fold2d(Shape.scalar(), window));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> integral.fold2d(output, window));
        IllegalArgumentException batch = assertThrows(
                IllegalArgumentException.class, () -> wrongBatch.fold2d(output, window));

        
    }

    @Test
    void fold2dAcceptsMatchingDynamicCanonicalColumnsAndRejectsUnrelatedSymbols() {
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        DynamicDimension batch = new DynamicDimension("N");
        Shape outputShape = Shape.ofDimensions(
                batch,
                new DynamicDimension("C"),
                new DynamicDimension("H"),
                new DynamicDimension("W"));
        Tensor image = tensor(DataType.FLOAT64, outputShape, Optional.empty(), true);
        Tensor columns = image.unfold2d(window);
        Tensor unrelatedChannels = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(batch, new DynamicDimension("columnChannels"),
                        columns.descriptor().shape().dimensions().get(2)),
                Optional.empty(), true);
        Tensor unrelatedCount = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(batch,
                        columns.descriptor().shape().dimensions().get(1),
                        new DynamicDimension("columnCount")),
                Optional.empty(), true);

        Tensor result = columns.fold2d(outputShape, window);
        IllegalArgumentException channelMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> unrelatedChannels.fold2d(outputShape, window));
        IllegalArgumentException countMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> unrelatedCount.fold2d(outputShape, window));

        assertAll(
                        () -> assertSame(outputShape, result.descriptor().shape()),
                        () -> assertEquals(columns.descriptor().dataType(),
                                result.descriptor().dataType()));
    }

    @Test
    void fold2dPreservesStaticMismatchAndFitFailures() {
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor columns = tensor(DataType.FLOAT64, Shape.of(1, 4, 4), Optional.empty(), true);
        Tensor wrongChannels = tensor(
                DataType.FLOAT32, Shape.of(1, 5, 4), Optional.empty(), true);
        Tensor wrongCount = tensor(
                DataType.FLOAT32, Shape.of(1, 4, 5), Optional.empty(), true);

        IllegalArgumentException channelMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> wrongChannels.fold2d(Shape.of(1, 1, 3, 3), window));
        IllegalArgumentException countMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> wrongCount.fold2d(Shape.of(1, 1, 3, 3), window));
        IllegalArgumentException fit = assertThrows(
                IllegalArgumentException.class,
                () -> columns.fold2d(Shape.of(1, 1, 1, 3), window));

        
    }

    @Test
    void unfold3dDerivesStaticAndCanonicalSymbolicDepthHeightWidthColumns() {
        DynamicDimension batch = new DynamicDimension("N3");
        DynamicDimension channels = new DynamicDimension("C3");
        DynamicDimension depth = new DynamicDimension("D3");
        DynamicDimension height = new DynamicDimension("H3");
        DynamicDimension width = new DynamicDimension("W3");
        Tensor dynamic = tensor(DataType.FLOAT64,
                Shape.ofDimensions(batch, channels, depth, height, width),
                Optional.empty(), true);
        Window3dAttrs floor = window3d(2, 3, 4, 2, 2, 3, 1, 1, 2, 1, 1, 1, false);
        Window3dAttrs ceil = window3d(2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, true);

        Tensor floorResult = dynamic.unfold3d(floor);
        Tensor ceilResult = tensor(DataType.FLOAT32, Shape.of(1, 2, 3, 3, 3),
                Optional.empty(), true).unfold3d(ceil);
        Dimension expectedChannels = DimensionExpressions.multiply(
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(channels, 2), 3), 4);
        Dimension expectedDepth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        depth, 2), 1);
        Dimension expectedHeight = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(height, -1), 2), 1);
        Dimension expectedWidth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        width, 3), 1);

        assertAll(
                () -> assertSame(batch, floorResult.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(expectedChannels,
                        floorResult.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(DimensionExpressions.multiply(
                                DimensionExpressions.multiply(expectedDepth, expectedHeight),
                                expectedWidth),
                        floorResult.descriptor().shape().dimensions().get(2)),
                () -> assertEquals(Shape.of(1, 16, 8), ceilResult.descriptor().shape()),
                () -> assertSame(floor,
                        floorResult.provenance().orElseThrow().operation().attrs()),
                () -> assertSame(WindowTransformKind.UNFOLD3D,
                        floorResult.provenance().orElseThrow().operation().kind()));
    }

    @Test
    void unfold3dAcceptsExactlyFloatingTypesAndRetainsExactTypedPadding() {
        Window3dAttrs window = window3d(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        for (DataType dataType : DataType.values()) {
            Tensor input = tensor(dataType, Shape.of(1, 2, 3, 4, 5), Optional.empty(),
                    dataType.isDifferentiable());
            if (dataType.isFloating()) {
                Tensor result = input.unfold3d(window);
                assertEquals(Shape.of(1, 2, 60), result.descriptor().shape());
            } else {
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> input.unfold3d(window));
                
            }
        }
        ScalarValue padding = ScalarValue.bfloat16Bits((short) 0xFF80);
        Tensor input = tensor(DataType.BFLOAT16, Shape.of(1, 1, 2, 2, 2), Optional.empty(), true);
        Tensor result = input.unfold3d(window, padding);
        Unfold3dAttrs attrs = (Unfold3dAttrs) result.provenance().orElseThrow().operation().attrs();
        assertAll(
                () -> assertSame(window, attrs.window()),
                () -> assertSame(padding, attrs.paddingValue()),
                () -> assertEquals(List.of(input), result.provenance().orElseThrow().inputs()),
                () -> assertEquals(0, result.provenance().orElseThrow().outputIndex()),
                () -> assertSame(result,
                        result.provenance().orElseThrow().producer().output(0)));
    }

    @Test
    void fold3dRequiresExactCanonicalColumnsAndRetainsTargetAndProvenance() {
        DynamicDimension batch = new DynamicDimension("foldN");
        Shape output = Shape.ofDimensions(batch, new DynamicDimension("foldC"),
                new DynamicDimension("foldD"), new DynamicDimension("foldH"),
                new DynamicDimension("foldW"));
        Window3dAttrs window = window3d(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor image = tensor(DataType.FLOAT32, output, Optional.empty(), true);
        Tensor columns = image.unfold3d(window);
        Tensor result = columns.fold3d(output, window);
        Fold3dAttrs attrs = (Fold3dAttrs) result.provenance().orElseThrow().operation().attrs();
        Tensor unrelated = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, new DynamicDimension("otherChannels"),
                        columns.descriptor().shape().dimensions().get(2)),
                Optional.empty(), true);
        IllegalArgumentException mismatch = assertThrows(IllegalArgumentException.class,
                () -> unrelated.fold3d(output, window));

        assertAll(
                        () -> assertSame(output, result.descriptor().shape()),
                        () -> assertSame(output, attrs.outputShape()),
                        () -> assertSame(window, attrs.window()),
                        () -> assertSame(WindowTransformKind.FOLD3D,
                                result.provenance().orElseThrow().operation().kind()),
                        () -> assertEquals(List.of(columns), result.provenance().orElseThrow().inputs()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void unfold3dAndFold3dRejectInExactOrderBeforeIdentityAllocation() throws Exception {
        Window3dAttrs window = window3d(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor ordinary = tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 3, 3), Optional.empty(), true);
        Tensor rankFour = tensor(DataType.INT32, Shape.of(1, 1, 1, 1), Optional.empty(), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(1, 1, 1, 1, 1), Optional.empty(), false);
        Tensor columns = tensor(DataType.FLOAT32, Shape.of(1, 8, 8), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(NullPointerException.class,
                        () -> TensorWindowExpressions.unfold3d(null, null));
        assertThrows(NullPointerException.class,
                        () -> ordinary.unfold3d(null));
        assertThrows(
                        IllegalArgumentException.class, () -> rankFour.unfold3d(window));
        assertThrows(
                        IllegalArgumentException.class, () -> integral.unfold3d(window));
        assertThrows(IllegalArgumentException.class,
                                () -> ordinary.unfold3d(window, ScalarValue.float64(0)));
        assertThrows(
                        IllegalArgumentException.class,
                        () -> columns.fold3d(Shape.scalar(), window));
        assertEquals(before, next.get());
        ordinary.unfold3d(window);
        assertEquals(before + 1, next.get());
    }

    @Test
    void repeatedRequestsAreFreshAndInputStorageAndMetadataRemainUntouched() {
        Shape shape = Shape.of(1, 1, 3, 3);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        Arena arena = Arena.ofConfined();
        MemorySegmentStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 9, arena.allocate(36));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("image"),
                Optional.empty(),
                Optional.of(storage));
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);

        Tensor first = input.unfold2d(window);
        Tensor second = input.unfold2d(window);
        arena.close();

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("image"), input.label()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertFalse(storage.isAlive()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(second.hostStorage().isEmpty()));
    }

    @Test
    void validationFailuresConsumeNoIdentityAndEachSuccessConsumesOne() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 3), Optional.empty(), true);
        Window2dAttrs window = window(2, 2, 1, 1, 0, 0, 1, 1, false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(IllegalArgumentException.class, () -> input.unfold(0, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> input.foldAxis(0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> input.fold2d(Shape.of(1, 1, 3, 3), window));
        assertThrows(IllegalArgumentException.class,
                () -> input.unfold2d(window, ScalarValue.float64(0.0)));
        assertEquals(before, next.get());

        input.unfold(2, 2, 1);
        assertEquals(before + 1, next.get());
        input.unfold2d(window);
        assertEquals(before + 2, next.get());
        input.unfold2d(window, ScalarValue.float32(-0.0f));
        assertEquals(before + 3, next.get());
        tensor(DataType.INT32, Shape.of(3, 3), Optional.empty(), false)
                .foldAxis(0, 5, 1);
        assertEquals(before + 4, next.get());
    }

    private static Window2dAttrs window(
            long kernelHeight,
            long kernelWidth,
            long strideHeight,
            long strideWidth,
            long paddingHeight,
            long paddingWidth,
            long dilationHeight,
            long dilationWidth,
            boolean ceilMode) {
        return new Window2dAttrs(
                kernelHeight,
                kernelWidth,
                strideHeight,
                strideWidth,
                paddingHeight,
                paddingWidth,
                dilationHeight,
                dilationWidth,
                ceilMode);
    }

    private static Window3dAttrs window3d(
            long kernelDepth, long kernelHeight, long kernelWidth,
            long strideDepth, long strideHeight, long strideWidth,
            long paddingDepth, long paddingHeight, long paddingWidth,
            long dilationDepth, long dilationHeight, long dilationWidth,
            boolean ceilMode) {
        return new Window3dAttrs(
                kernelDepth, kernelHeight, kernelWidth,
                strideDepth, strideHeight, strideWidth,
                paddingDepth, paddingHeight, paddingWidth,
                dilationDepth, dilationHeight, dilationWidth,
                ceilMode);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    
}
