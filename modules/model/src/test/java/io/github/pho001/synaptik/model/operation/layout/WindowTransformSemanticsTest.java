package io.github.pho001.synaptik.model.operation.layout;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class WindowTransformSemanticsTest {
    @Test
    void declaresExactlyTheFourOrderedWindowTransformKinds() {
        assertAll(
                () -> assertArrayEquals(
                        new WindowTransformKind[] {
                            WindowTransformKind.UNFOLD_AXIS,
                            WindowTransformKind.FOLD_AXIS,
                            WindowTransformKind.UNFOLD2D,
                            WindowTransformKind.FOLD2D
                        },
                        WindowTransformKind.values()),
                () -> assertEquals("UNFOLD_AXIS", WindowTransformKind.UNFOLD_AXIS.name()),
                () -> assertEquals("FOLD_AXIS", WindowTransformKind.FOLD_AXIS.name()),
                () -> assertEquals("UNFOLD2D", WindowTransformKind.UNFOLD2D.name()),
                () -> assertEquals("FOLD2D", WindowTransformKind.FOLD2D.name()),
                () -> assertSame(
                        WindowTransformKind.UNFOLD_AXIS,
                        WindowTransformKind.valueOf("UNFOLD_AXIS")),
                () -> assertSame(
                        WindowTransformKind.FOLD_AXIS,
                        WindowTransformKind.valueOf("FOLD_AXIS")),
                () -> assertSame(
                        WindowTransformKind.UNFOLD2D,
                        WindowTransformKind.valueOf("UNFOLD2D")),
                () -> assertSame(
                        WindowTransformKind.FOLD2D,
                        WindowTransformKind.valueOf("FOLD2D")),
                () -> assertInstanceOf(
                        OperationKind.class, WindowTransformKind.UNFOLD_AXIS));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(WindowTransformKind.class);
    }

    @Test
    void exposesOnlyTheExactRecordShapes() {
        assertRecordShape(
                UnfoldAxisAttrs.class,
                List.of("axis", "size", "step"),
                List.of(int.class, long.class, long.class),
                List.of(
                        "axis():int",
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "size():long",
                        "step():long",
                        "toString():java.lang.String"));
        assertRecordShape(
                FoldAxisAttrs.class,
                List.of("axis", "outputSize", "step"),
                List.of(int.class, long.class, long.class),
                List.of(
                        "axis():int",
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "outputSize():long",
                        "step():long",
                        "toString():java.lang.String"));
        assertRecordShape(
                Window2dAttrs.class,
                List.of(
                        "kernelHeight",
                        "kernelWidth",
                        "strideHeight",
                        "strideWidth",
                        "paddingHeight",
                        "paddingWidth",
                        "dilationHeight",
                        "dilationWidth",
                        "ceilMode"),
                List.of(
                        long.class,
                        long.class,
                        long.class,
                        long.class,
                        long.class,
                        long.class,
                        long.class,
                        long.class,
                        boolean.class),
                List.of(
                        "ceilMode():boolean",
                        "dilationHeight():long",
                        "dilationWidth():long",
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "kernelHeight():long",
                        "kernelWidth():long",
                        "paddingHeight():long",
                        "paddingWidth():long",
                        "strideHeight():long",
                        "strideWidth():long",
                        "toString():java.lang.String"));
        assertRecordShape(
                Unfold2dAttrs.class,
                List.of("window", "paddingValue"),
                List.of(Window2dAttrs.class, ScalarValue.class),
                List.of(
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "paddingValue():io.github.pho001.synaptik.model.datatype.ScalarValue",
                        "toString():java.lang.String",
                        "window():io.github.pho001.synaptik.model.operation.layout.Window2dAttrs"));
        assertRecordShape(
                Fold2dAttrs.class,
                List.of("outputShape", "window"),
                List.of(Shape.class, Window2dAttrs.class),
                List.of(
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "outputShape():io.github.pho001.synaptik.model.shape.Shape",
                        "toString():java.lang.String",
                        "window():io.github.pho001.synaptik.model.operation.layout.Window2dAttrs"));
    }

    @Test
    void retainsOrdinaryAndExtremeSingleAxisValues() {
        UnfoldAxisAttrs unfold = new UnfoldAxisAttrs(1, 3, 2);
        UnfoldAxisAttrs extremeUnfold =
                new UnfoldAxisAttrs(Integer.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);
        FoldAxisAttrs fold = new FoldAxisAttrs(0, 5, 1);
        FoldAxisAttrs zeroFold = new FoldAxisAttrs(0, 0, 1);
        FoldAxisAttrs extremeFold =
                new FoldAxisAttrs(Integer.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);

        assertAll(
                () -> assertEquals(1, unfold.axis()),
                () -> assertEquals(3, unfold.size()),
                () -> assertEquals(2, unfold.step()),
                () -> assertEquals(Integer.MAX_VALUE, extremeUnfold.axis()),
                () -> assertEquals(Long.MAX_VALUE, extremeUnfold.size()),
                () -> assertEquals(Long.MAX_VALUE, extremeUnfold.step()),
                () -> assertEquals(0, fold.axis()),
                () -> assertEquals(5, fold.outputSize()),
                () -> assertEquals(1, fold.step()),
                () -> assertEquals(0, zeroFold.outputSize()),
                () -> assertEquals(Integer.MAX_VALUE, extremeFold.axis()),
                () -> assertEquals(Long.MAX_VALUE, extremeFold.outputSize()),
                () -> assertEquals(Long.MAX_VALUE, extremeFold.step()));
    }

    @Test
    void validatesUnfoldAxisComponentsInExactOrderWithExactMessages() {
        assertAll(
                () -> assertIllegalFailure(
                        () -> new UnfoldAxisAttrs(-1, 0, 0),
                        "axis must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new UnfoldAxisAttrs(0, 0, 0), "size must be positive: 0"),
                () -> assertIllegalFailure(
                        () -> new UnfoldAxisAttrs(0, Long.MIN_VALUE, -1),
                        "size must be positive: " + Long.MIN_VALUE),
                () -> assertIllegalFailure(
                        () -> new UnfoldAxisAttrs(0, 1, 0), "step must be positive: 0"),
                () -> assertIllegalFailure(
                        () -> new UnfoldAxisAttrs(0, 1, Long.MIN_VALUE),
                        "step must be positive: " + Long.MIN_VALUE));
    }

    @Test
    void validatesFoldAxisComponentsInExactOrderWithExactMessages() {
        assertAll(
                () -> assertIllegalFailure(
                        () -> new FoldAxisAttrs(-1, -1, 0),
                        "axis must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new FoldAxisAttrs(0, -1, 0),
                        "outputSize must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new FoldAxisAttrs(0, Long.MIN_VALUE, -1),
                        "outputSize must be non-negative: " + Long.MIN_VALUE),
                () -> assertIllegalFailure(
                        () -> new FoldAxisAttrs(0, 0, 0), "step must be positive: 0"),
                () -> assertIllegalFailure(
                        () -> new FoldAxisAttrs(0, 1, Long.MIN_VALUE),
                        "step must be positive: " + Long.MIN_VALUE));
    }

    @Test
    void retainsEveryWindow2dComponentWithoutArithmetic() {
        Window2dAttrs ordinary = new Window2dAttrs(2, 3, 4, 5, 0, 6, 7, 8, true);
        Window2dAttrs extreme = new Window2dAttrs(
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                false);

        assertAll(
                () -> assertEquals(2, ordinary.kernelHeight()),
                () -> assertEquals(3, ordinary.kernelWidth()),
                () -> assertEquals(4, ordinary.strideHeight()),
                () -> assertEquals(5, ordinary.strideWidth()),
                () -> assertEquals(0, ordinary.paddingHeight()),
                () -> assertEquals(6, ordinary.paddingWidth()),
                () -> assertEquals(7, ordinary.dilationHeight()),
                () -> assertEquals(8, ordinary.dilationWidth()),
                () -> assertTrue(ordinary.ceilMode()),
                () -> assertEquals(Long.MAX_VALUE, extreme.kernelHeight()),
                () -> assertEquals(Long.MAX_VALUE, extreme.paddingWidth()),
                () -> assertEquals(Long.MAX_VALUE, extreme.dilationWidth()),
                () -> assertFalse(extreme.ceilMode()));
    }

    @Test
    void validatesWindow2dComponentsInExactOrderWithExactMessages() {
        assertWindowFailure(
                new long[] {0, 0, 0, 0, -1, -1, 0, 0},
                "kernelHeight must be positive: 0");
        assertWindowFailure(
                new long[] {1, -2, 0, 0, -1, -1, 0, 0},
                "kernelWidth must be positive: -2");
        assertWindowFailure(
                new long[] {1, 1, 0, 0, -1, -1, 0, 0},
                "strideHeight must be positive: 0");
        assertWindowFailure(
                new long[] {1, 1, 1, -4, -1, -1, 0, 0},
                "strideWidth must be positive: -4");
        assertWindowFailure(
                new long[] {1, 1, 1, 1, -5, -1, 0, 0},
                "paddingHeight must be non-negative: -5");
        assertWindowFailure(
                new long[] {1, 1, 1, 1, 0, -6, 0, 0},
                "paddingWidth must be non-negative: -6");
        assertWindowFailure(
                new long[] {1, 1, 1, 1, 0, 0, 0, 0},
                "dilationHeight must be positive: 0");
        assertWindowFailure(
                new long[] {1, 1, 1, 1, 0, 0, 1, -8},
                "dilationWidth must be positive: -8");
    }

    @Test
    void fold2dNullChecksInOrderAndRetainsExactReferencesForEveryShapeCategory() {
        Window2dAttrs window = unitWindow();
        Shape scalar = Shape.scalar();
        Shape nonRankFour = Shape.of(2, 3);
        Shape zeroExtent = Shape.of(1, 1, 0, 3);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));

        assertNullFailure(() -> new Fold2dAttrs(null, null), "outputShape");
        assertNullFailure(() -> new Fold2dAttrs(scalar, null), "window");

        for (Shape shape : List.of(scalar, nonRankFour, zeroExtent, dynamic)) {
            Fold2dAttrs attrs = new Fold2dAttrs(shape, window);
            assertAll(
                    () -> assertSame(shape, attrs.outputShape()),
                    () -> assertSame(window, attrs.window()));
        }
    }

    @Test
    void unfold2dAttrsNullChecksInOrderAndRetainsExactTypedReferences() {
        Window2dAttrs window = unitWindow();
        List<ScalarValue> values = List.of(
                ScalarValue.float64(Double.longBitsToDouble(0xFFF8_0000_0000_0042L)),
                ScalarValue.float32(Float.intBitsToFloat(0x8000_0000)),
                ScalarValue.bfloat16Bits((short) 0xFFC1),
                ScalarValue.int32(Integer.MIN_VALUE),
                ScalarValue.int64(Long.MIN_VALUE),
                ScalarValue.bool(true));

        assertNullFailure(() -> new Unfold2dAttrs(null, null), "window");
        assertNullFailure(() -> new Unfold2dAttrs(window, null), "paddingValue");
        for (ScalarValue value : values) {
            Unfold2dAttrs attrs = new Unfold2dAttrs(window, value);
            assertAll(
                    () -> assertSame(window, attrs.window()),
                    () -> assertSame(value, attrs.paddingValue()),
                    () -> assertEquals(attrs, new Unfold2dAttrs(window, value)),
                    () -> assertEquals(attrs.hashCode(),
                            new Unfold2dAttrs(window, value).hashCode()));
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        UnfoldAxisAttrs unfold = new UnfoldAxisAttrs(1, 3, 2);
        FoldAxisAttrs fold = new FoldAxisAttrs(0, 5, 1);
        Window2dAttrs window = unitWindow();
        Shape outputShape = Shape.of(1, 1, 3, 3);
        Fold2dAttrs fold2d = new Fold2dAttrs(outputShape, window);

        assertAll(
                () -> assertEquals(unfold, new UnfoldAxisAttrs(1, 3, 2)),
                () -> assertEquals(unfold.hashCode(), new UnfoldAxisAttrs(1, 3, 2).hashCode()),
                () -> assertNotEquals(unfold, new UnfoldAxisAttrs(0, 3, 2)),
                () -> assertEquals("UnfoldAxisAttrs[axis=1, size=3, step=2]", unfold.toString()),
                () -> assertEquals(fold, new FoldAxisAttrs(0, 5, 1)),
                () -> assertNotEquals(fold, new FoldAxisAttrs(0, 4, 1)),
                () -> assertEquals(
                        "FoldAxisAttrs[axis=0, outputSize=5, step=1]", fold.toString()),
                () -> assertEquals(window, unitWindow()),
                () -> assertNotEquals(window, new Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, true)),
                () -> assertEquals(fold2d, new Fold2dAttrs(Shape.of(1, 1, 3, 3), unitWindow())),
                () -> assertEquals(fold2d.hashCode(),
                        new Fold2dAttrs(Shape.of(1, 1, 3, 3), unitWindow()).hashCode()));
    }

    @Test
    void composesExactKindsWithExactAttributeReferences() {
        UnfoldAxisAttrs unfoldAttrs = new UnfoldAxisAttrs(1, 3, 1);
        FoldAxisAttrs foldAttrs = new FoldAxisAttrs(0, 5, 1);
        Window2dAttrs windowAttrs = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        Fold2dAttrs fold2dAttrs = new Fold2dAttrs(Shape.of(1, 1, 3, 3), windowAttrs);
        Unfold2dAttrs explicitUnfoldAttrs =
                new Unfold2dAttrs(windowAttrs, ScalarValue.float32(-0.0f));
        Operation axisUnfold =
                new Operation(WindowTransformKind.UNFOLD_AXIS, unfoldAttrs);
        Operation axisFold = new Operation(WindowTransformKind.FOLD_AXIS, foldAttrs);
        Operation imageUnfold = new Operation(WindowTransformKind.UNFOLD2D, windowAttrs);
        Operation explicitImageUnfold =
                new Operation(WindowTransformKind.UNFOLD2D, explicitUnfoldAttrs);
        Operation imageFold = new Operation(WindowTransformKind.FOLD2D, fold2dAttrs);

        assertAll(
                () -> assertSame(WindowTransformKind.UNFOLD_AXIS, axisUnfold.kind()),
                () -> assertSame(unfoldAttrs, axisUnfold.attrs()),
                () -> assertSame(WindowTransformKind.FOLD_AXIS, axisFold.kind()),
                () -> assertSame(foldAttrs, axisFold.attrs()),
                () -> assertSame(WindowTransformKind.UNFOLD2D, imageUnfold.kind()),
                () -> assertSame(windowAttrs, imageUnfold.attrs()),
                () -> assertSame(WindowTransformKind.UNFOLD2D, explicitImageUnfold.kind()),
                () -> assertSame(explicitUnfoldAttrs, explicitImageUnfold.attrs()),
                () -> assertSame(WindowTransformKind.FOLD2D, imageFold.kind()),
                () -> assertSame(fold2dAttrs, imageFold.attrs()));
    }

    @Test
    void attributesContainOnlySpecifiedStateWithoutForbiddenDependencies() {
        List<String> unfoldTypes = componentTypeNames(UnfoldAxisAttrs.class);
        List<String> foldTypes = componentTypeNames(FoldAxisAttrs.class);
        List<String> windowTypes = componentTypeNames(Window2dAttrs.class);
        List<String> explicitUnfoldTypes = componentTypeNames(Unfold2dAttrs.class);
        List<String> fold2dTypes = componentTypeNames(Fold2dAttrs.class);

        assertAll(
                () -> assertEquals(List.of("int", "long", "long"), unfoldTypes),
                () -> assertEquals(List.of("int", "long", "long"), foldTypes),
                () -> assertEquals(
                        List.of(
                                "long", "long", "long", "long", "long", "long", "long",
                                "long", "boolean"),
                        windowTypes),
                () -> assertEquals(
                        List.of(
                                "io.github.pho001.synaptik.model.operation.layout.Window2dAttrs",
                                "io.github.pho001.synaptik.model.datatype.ScalarValue"),
                        explicitUnfoldTypes),
                () -> assertEquals(
                        List.of(
                                "io.github.pho001.synaptik.model.shape.Shape",
                                "io.github.pho001.synaptik.model.operation.layout.Window2dAttrs"),
                        fold2dTypes),
                () -> assertFalse(unfoldTypes.stream().anyMatch(
                        WindowTransformSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(foldTypes.stream().anyMatch(
                        WindowTransformSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(windowTypes.stream().anyMatch(
                        WindowTransformSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(fold2dTypes.stream().anyMatch(
                        WindowTransformSemanticsTest::isForbiddenComponentType)));
    }

    private static void assertRecordShape(
            Class<?> recordClass,
            List<String> componentNames,
            List<Class<?>> componentTypes,
            List<String> expectedMethods) {
        var components = recordClass.getRecordComponents();
        var constructors = recordClass.getDeclaredConstructors();
        var fields = recordClass.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        recordClass.getPackageName()),
                () -> assertTrue(Modifier.isPublic(recordClass.getModifiers())),
                () -> assertTrue(Modifier.isFinal(recordClass.getModifiers())),
                () -> assertTrue(recordClass.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(recordClass.getInterfaces())),
                () -> assertEquals(
                        componentNames,
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        componentTypes,
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        componentTypes, Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        componentNames,
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        expectedMethods,
                        Arrays.stream(recordClass.getDeclaredMethods())
                                .map(WindowTransformSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, recordClass.getDeclaredClasses().length));
    }

    private static void assertWindowFailure(long[] values, String expectedMessage) {
        assertIllegalFailure(
                () -> new Window2dAttrs(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        false),
                expectedMessage);
    }

    private static Window2dAttrs unitWindow() {
        return new Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
    }

    private static List<String> componentTypeNames(Class<?> recordClass) {
        return Arrays.stream(recordClass.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();
    }

    private static void assertNullFailure(
            org.junit.jupiter.api.function.Executable construction, String expectedMessage) {
        NullPointerException failure = assertThrows(NullPointerException.class, construction);
        assertEquals(expectedMessage, failure.getMessage());
    }

    private static void assertIllegalFailure(
            org.junit.jupiter.api.function.Executable construction, String expectedMessage) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, construction);
        assertEquals(expectedMessage, failure.getMessage());
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName()
                + "("
                + parameters
                + "):"
                + method.getReturnType().getName();
    }

    private static boolean isForbiddenComponentType(String name) {
        return name.contains("Tensor")
                || name.contains("DataType")
                || name.contains("layout.Layout")
                || name.contains("storage")
                || name.contains("provenance")
                || name.contains("graph")
                || name.contains("compiler")
                || name.contains("planning")
                || name.contains("prepare")
                || name.contains("runtime")
                || name.contains("backend")
                || name.contains("onnx")
                || name.contains("training");
    }
}
