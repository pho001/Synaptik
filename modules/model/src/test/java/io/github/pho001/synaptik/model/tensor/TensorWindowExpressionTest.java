package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorWindowExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(4_000_000);

    @Test
    void helperHasExactlyTheRequiredFieldFreeMethodShape() {
        var constructors = TensorWindowExpressions.class.getDeclaredConstructors();
        var methods = Arrays.stream(TensorWindowExpressions.class.getDeclaredMethods())
                .map(TensorWindowExpressionTest::methodSignature)
                .sorted()
                .toList();

        assertAll(
                () -> assertFalse(Modifier.isPublic(
                        TensorWindowExpressions.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        TensorWindowExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorWindowExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorWindowExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(
                        List.of(
                                "create(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.Operation):io.github.pho001.synaptik.model.tensor.Tensor",
                                "dimensionDiagnostic(io.github.pho001.synaptik.model.shape.Dimension):java.lang.String",
                                "fold2d(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.layout.Window2dAttrs):io.github.pho001.synaptik.model.tensor.Tensor",
                                "foldAxis(io.github.pho001.synaptik.model.tensor.Tensor,int,long,long):io.github.pho001.synaptik.model.tensor.Tensor",
                                "foldAxisShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs):io.github.pho001.synaptik.model.shape.Shape",
                                "normalizeAxis(int,int):int",
                                "requireStaticSize(io.github.pho001.synaptik.model.shape.Shape,int,java.lang.String,java.lang.String):long",
                                "unfold2d(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.operation.layout.Window2dAttrs):io.github.pho001.synaptik.model.tensor.Tensor",
                                "unfold2d(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.operation.layout.Window2dAttrs,io.github.pho001.synaptik.model.datatype.ScalarValue):io.github.pho001.synaptik.model.tensor.Tensor",
                                "unfold2dShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.layout.Window2dAttrs):io.github.pho001.synaptik.model.shape.Shape",
                                "unfoldAxis(io.github.pho001.synaptik.model.tensor.Tensor,int,long,long):io.github.pho001.synaptik.model.tensor.Tensor",
                                "unfoldAxisShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs):io.github.pho001.synaptik.model.shape.Shape",
                                "validateFloating(io.github.pho001.synaptik.model.datatype.DataType,java.lang.String):void",
                                "validateFold2dShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.operation.layout.Window2dAttrs):void",
                                "validateNumeric(io.github.pho001.synaptik.model.datatype.DataType,java.lang.String):void",
                                "windowOutputDimension(io.github.pho001.synaptik.model.shape.Dimension,long,long,long,long,boolean,java.lang.String,java.lang.String):io.github.pho001.synaptik.model.shape.Dimension",
                                "windowOutputSize(long,long,long,long,long,boolean,java.lang.String,java.lang.String):long"),
                        methods),
                () -> assertTrue(Arrays.stream(
                                TensorWindowExpressions.class.getDeclaredMethods())
                        .allMatch(method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(Arrays.stream(
                                TensorWindowExpressions.class.getDeclaredMethods())
                        .noneMatch(method -> method.isSynthetic() || method.isBridge())),
                () -> assertEquals(Tensor.class, Tensor.class.getDeclaredMethod(
                        "foldAxis", int.class, long.class, long.class).getReturnType()),
                () -> assertEquals(Tensor.class, Tensor.class.getDeclaredMethod(
                        "unfold2d", Window2dAttrs.class, ScalarValue.class).getReturnType()));
    }

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

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("unfold requires rank at least 1", rank.getMessage()),
                () -> assertEquals(
                        "Axis -2147483648 is outside shape rank 1", axis.getMessage()),
                () -> assertEquals("size must be positive: 0", size.getMessage()),
                () -> assertEquals("step must be positive: 0", step.getMessage()),
                () -> assertEquals(
                        "unfold requires static selected dimension at axis 0",
                        staticity.getMessage()),
                () -> assertEquals(
                        "unfold size 1 exceeds selected dimension 0", zeroFit.getMessage()),
                () -> assertEquals(
                        "unfold size 3 exceeds selected dimension 2", fit.getMessage()));
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
                () -> assertEquals(Shape.of(2, 0), result.descriptor().shape()),
                () -> assertEquals(
                        "foldAxis window count 1 does not match output size and window geometry: expected=0",
                        mismatch.getMessage()));
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

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("foldAxis requires rank at least 2", rank.getMessage()),
                () -> assertEquals("Axis -2147483648 is outside shape rank 1", axis.getMessage()),
                () -> assertEquals("outputSize must be non-negative: -1",
                        outputSize.getMessage()),
                () -> assertEquals("step must be positive: 0", step.getMessage()),
                () -> assertEquals("foldAxis requires floating or integral input: BOOL",
                        type.getMessage()),
                () -> assertEquals("foldAxis requires static window-count dimension at axis 0",
                        countStaticity.getMessage()),
                () -> assertEquals("foldAxis requires a positive static final window dimension",
                        windowStaticity.getMessage()),
                () -> assertEquals("foldAxis requires a positive static final window dimension",
                        positiveWindow.getMessage()),
                () -> assertEquals("foldAxis window size 6 exceeds output size 5",
                        windowFit.getMessage()),
                () -> assertEquals(
                        "foldAxis window count 2 does not match output size and window geometry: expected=3",
                        count.getMessage()));
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
                assertEquals(
                        "unfold2d requires floating input: " + dataType,
                        failure.getMessage());
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
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("window", nullWindow.getMessage()),
                () -> assertEquals("paddingValue", nullPadding.getMessage()),
                () -> assertEquals(
                        "unfold2d paddingValue data type must match input data type: paddingValue=FLOAT64, input=FLOAT32",
                        mismatch.getMessage()),
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

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("window", nullWindow.getMessage()),
                () -> assertEquals("unfold2d requires rank-4 NCHW input", rank.getMessage()),
                () -> assertEquals("unfold2d requires floating input: INT32", type.getMessage()),
                () -> assertEquals(
                        "unfold2d effective kernel does not fit padded height", fit.getMessage()),
                () -> assertTrue(dilationOverflow.getMessage().contains("long overflow")),
                () -> assertTrue(paddingOverflow.getMessage().contains("long overflow")));
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

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("outputShape", nullOutput.getMessage()),
                () -> assertEquals("window", nullWindow.getMessage()),
                () -> assertEquals(
                        "fold2d requires rank-3 canonical column input", inputRank.getMessage()),
                () -> assertEquals(
                        "fold2d outputShape must be rank-4 NCHW", outputRank.getMessage()),
                () -> assertEquals("fold2d requires floating input: INT32", type.getMessage()),
                () -> assertEquals(
                        "fold2d output batch dimension must match column batch dimension",
                        batch.getMessage()));
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
                        result.descriptor().dataType()),
                () -> assertTrue(channelMismatch.getMessage().startsWith(
                        "fold2d column-channel dimension ")),
                () -> assertTrue(channelMismatch.getMessage().contains(
                        " does not match output channels and kernel geometry: expected=")),
                () -> assertTrue(countMismatch.getMessage().startsWith(
                        "fold2d column count ")),
                () -> assertTrue(countMismatch.getMessage().contains(
                        " does not match output shape and window geometry: expected=")));
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

        assertAll(
                () -> assertEquals(
                        "fold2d column-channel dimension 5 does not match output channels and kernel geometry: expected=4",
                        channelMismatch.getMessage()),
                () -> assertEquals(
                        "fold2d column count 5 does not match output shape and window geometry: expected=4",
                        countMismatch.getMessage()),
                () -> assertEquals(
                        "fold2d effective kernel does not fit padded height", fit.getMessage()));
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

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .reduce((left, right) -> left + "," + right)
                .orElse("");
        return method.getName() + "(" + parameters + "):" + method.getReturnType().getName();
    }
}
