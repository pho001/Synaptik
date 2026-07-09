package io.github.pho001.synaptik.model.operation.layout;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class PadTileSemanticsTest {
    @Test
    void declaresExactlyTheSeparatePadAndTileKinds() {
        OperationKind pad = PadKind.PAD;
        OperationKind tile = TileKind.TILE;

        assertAll(
                () -> assertArrayEquals(new PadKind[] {PadKind.PAD}, PadKind.values()),
                () -> assertArrayEquals(new TileKind[] {TileKind.TILE}, TileKind.values()),
                () -> assertEquals("PAD", pad.name()),
                () -> assertEquals("TILE", tile.name()),
                () -> assertSame(PadKind.PAD, PadKind.valueOf("PAD")),
                () -> assertSame(TileKind.TILE, TileKind.valueOf("TILE")),
                () -> assertInstanceOf(OperationKind.class, pad),
                () -> assertInstanceOf(OperationKind.class, tile),
                () -> assertThrows(IllegalArgumentException.class, () -> PadKind.valueOf("TILE")),
                () -> assertThrows(IllegalArgumentException.class, () -> TileKind.valueOf("PAD")));
    }

    @Test
    void exposesOnlyTheExactEnumShapes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(PadKind.class);
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(TileKind.class);
    }

    @Test
    void exposesOnlyTheExactRecordShapes() {
        var padComponents = PadAttrs.class.getRecordComponents();
        var padConstructors = PadAttrs.class.getDeclaredConstructors();
        var padFields = PadAttrs.class.getDeclaredFields();
        var tileComponents = TileAttrs.class.getRecordComponents();
        var tileConstructors = TileAttrs.class.getDeclaredConstructors();
        var tileFields = TileAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertRecordType(PadAttrs.class),
                () -> assertEquals(
                        List.of("before", "after", "constantValue"),
                        Arrays.stream(padComponents).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(List.class, List.class, ScalarValue.class),
                        Arrays.stream(padComponents).map(component -> component.getType()).toList()),
                () -> assertEquals(
                        List.of(
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Long>",
                                "io.github.pho001.synaptik.model.datatype.ScalarValue"),
                        Arrays.stream(padComponents)
                                .map(component -> component.getGenericType().getTypeName())
                                .toList()),
                () -> assertEquals(1, padConstructors.length),
                () -> assertEquals(
                        List.of(List.class, List.class, ScalarValue.class),
                        Arrays.asList(padConstructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(padConstructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("before", "after", "constantValue"),
                        Arrays.stream(padFields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(padFields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "after():java.util.List",
                                "before():java.util.List",
                                "constantValue():io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        declaredMethodSignatures(PadAttrs.class)),
                () -> assertEquals(0, PadAttrs.class.getDeclaredClasses().length),
                () -> assertRecordType(TileAttrs.class),
                () -> assertEquals(
                        List.of("repeats"),
                        Arrays.stream(tileComponents)
                                .map(component -> component.getName())
                                .toList()),
                () -> assertEquals(
                        List.of(List.class),
                        Arrays.stream(tileComponents).map(component -> component.getType()).toList()),
                () -> assertEquals(
                        List.of("java.util.List<java.lang.Long>"),
                        Arrays.stream(tileComponents)
                                .map(component -> component.getGenericType().getTypeName())
                                .toList()),
                () -> assertEquals(1, tileConstructors.length),
                () -> assertEquals(
                        List.of(List.class),
                        Arrays.asList(tileConstructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(tileConstructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("repeats"),
                        Arrays.stream(tileFields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(tileFields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "repeats():java.util.List",
                                "toString():java.lang.String"),
                        declaredMethodSignatures(TileAttrs.class)),
                () -> assertEquals(0, TileAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsScalarIdentityOrdinaryExamplesAndExtremeWidths() {
        PadAttrs scalarPad = new PadAttrs(List.of(), List.of(), v(-0.0d));
        TileAttrs scalarTile = new TileAttrs(List.of());
        PadAttrs pad = new PadAttrs(List.of(1L), List.of(2L), v(-1.0d));
        TileAttrs tile = new TileAttrs(List.of(2L, 3L));
        PadAttrs extremePad =
                new PadAttrs(List.of(Long.MAX_VALUE), List.of(Long.MAX_VALUE), v(1.0d));
        TileAttrs extremeTile = new TileAttrs(List.of(Long.MAX_VALUE));

        assertAll(
                () -> assertEquals(List.of(), scalarPad.before()),
                () -> assertEquals(List.of(), scalarPad.after()),
                () -> assertEquals(
                        Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(scalarPad.constantValue().float64Value())),
                () -> assertEquals(List.of(), scalarTile.repeats()),
                () -> assertEquals(List.of(1L), pad.before()),
                () -> assertEquals(List.of(2L), pad.after()),
                () -> assertEquals(v(-1.0d), pad.constantValue()),
                () -> assertEquals(List.of(2L, 3L), tile.repeats()),
                () -> assertEquals(List.of(Long.MAX_VALUE), extremePad.before()),
                () -> assertEquals(List.of(Long.MAX_VALUE), extremePad.after()),
                () -> assertEquals(List.of(Long.MAX_VALUE), extremeTile.repeats()));
    }

    @Test
    void snapshotsCallerListsAndExposesImmutableValues() {
        ArrayList<Long> callerBefore = new ArrayList<>(List.of(1L, 0L));
        ArrayList<Long> callerAfter = new ArrayList<>(List.of(2L, 3L));
        ArrayList<Long> callerRepeats = new ArrayList<>(List.of(2L, 3L));
        ScalarValue constant = v(4.5d);
        PadAttrs pad = new PadAttrs(callerBefore, callerAfter, constant);
        TileAttrs tile = new TileAttrs(callerRepeats);

        callerBefore.set(0, 7L);
        callerAfter.set(0, 8L);
        callerRepeats.set(0, 9L);
        callerBefore.add(1L);
        callerAfter.add(1L);
        callerRepeats.add(1L);

        assertAll(
                () -> assertEquals(List.of(1L, 0L), pad.before()),
                () -> assertEquals(List.of(2L, 3L), pad.after()),
                () -> assertEquals(List.of(2L, 3L), tile.repeats()),
                () -> assertSame(constant, pad.constantValue()),
                () -> assertNotSame(callerBefore, pad.before()),
                () -> assertNotSame(callerAfter, pad.after()),
                () -> assertNotSame(callerRepeats, tile.repeats()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> pad.before().set(0, 5L)),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> pad.after().set(0, 5L)),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> tile.repeats().set(0, 5L)));
    }

    @Test
    void retainsEveryDoubleConstantWithoutValidationOrNormalization() {
        long[] rawBits = {
            Double.doubleToRawLongBits(3.25d),
            Double.doubleToRawLongBits(0.0d),
            Double.doubleToRawLongBits(-0.0d),
            Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
            Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
            0x7ff8_0000_0000_0001L,
            0xfff8_0000_0000_0042L
        };

        for (long bits : rawBits) {
            double supplied = Double.longBitsToDouble(bits);
            PadAttrs attrs = new PadAttrs(List.of(), List.of(), ScalarValue.float64(supplied));

            assertEquals(bits, Double.doubleToRawLongBits(attrs.constantValue().float64Value()));
        }
    }

    @Test
    void rejectsPadNullContainersAndMismatchedSizesWithExactPrecedence() {
        assertNullFailure(() -> new PadAttrs(null, null, null), "before");
        assertNullFailure(() -> new PadAttrs(List.of(), null, null), "after");
        assertNullFailure(() -> new PadAttrs(List.of(), List.of(0L), null), "constantValue");

        assertAll(
                () -> assertIllegalFailure(
                        () -> new PadAttrs(List.of(), List.of(0L), v(0.0d)),
                        "before and after must have matching sizes"),
                () -> assertIllegalFailure(
                        () -> new PadAttrs(List.of(0L), List.of(), v(0.0d)),
                        "before and after must have matching sizes"),
                () -> assertIllegalFailure(
                        () -> new PadAttrs(
                                Arrays.asList((Long) null), List.of(), v(Double.NaN)),
                        "before and after must have matching sizes"));
    }

    @Test
    void rejectsPadElementsInExactIndexAndValidationOrder() {
        assertNullFailure(
                () -> new PadAttrs(
                        Arrays.asList((Long) null), Arrays.asList((Long) null), v(0.0d)),
                "before[0]");
        assertNullFailure(
                () -> new PadAttrs(List.of(-1L), Arrays.asList((Long) null), v(0.0d)),
                "after[0]");
        assertIllegalFailure(
                () -> new PadAttrs(List.of(-1L), List.of(-2L), v(0.0d)),
                "before[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new PadAttrs(List.of(0L), List.of(-2L), v(0.0d)),
                "after[0] must be non-negative: -2");
        assertIllegalFailure(
                () -> new PadAttrs(
                        List.of(-1L, 0L), Arrays.asList(0L, null), v(0.0d)),
                "before[0] must be non-negative: -1");
        assertNullFailure(
                () -> new PadAttrs(
                        Arrays.asList(0L, null), Arrays.asList(0L, null), v(0.0d)),
                "before[1]");
        assertNullFailure(
                () -> new PadAttrs(List.of(0L, 0L), Arrays.asList(0L, null), v(0.0d)),
                "after[1]");
    }

    @Test
    void rejectsTileNullAndNonPositiveElementsWithExactPrecedence() {
        assertNullFailure(() -> new TileAttrs(null), "repeats");
        assertNullFailure(() -> new TileAttrs(Arrays.asList((Long) null)), "repeats[0]");
        assertNullFailure(() -> new TileAttrs(Arrays.asList(1L, null)), "repeats[1]");
        assertIllegalFailure(
                () -> new TileAttrs(List.of(0L)), "repeats[0] must be positive: 0");
        assertIllegalFailure(
                () -> new TileAttrs(List.of(-3L)), "repeats[0] must be positive: -3");
        assertIllegalFailure(
                () -> new TileAttrs(Arrays.asList(0L, null)),
                "repeats[0] must be positive: 0");
    }

    @Test
    void usesOrderedRecordValueSemanticsAndDiagnosticText() {
        PadAttrs pad = new PadAttrs(List.of(1L, 0L), List.of(2L, 3L), v(-1.0d));
        PadAttrs equalPad = new PadAttrs(List.of(1L, 0L), List.of(2L, 3L), v(-1.0d));
        PadAttrs reorderedPad = new PadAttrs(List.of(0L, 1L), List.of(3L, 2L), v(-1.0d));
        TileAttrs tile = new TileAttrs(List.of(2L, 3L));
        TileAttrs equalTile = new TileAttrs(List.of(2L, 3L));
        TileAttrs reorderedTile = new TileAttrs(List.of(3L, 2L));

        assertAll(
                () -> assertEquals(pad, equalPad),
                () -> assertEquals(pad.hashCode(), equalPad.hashCode()),
                () -> assertNotEquals(pad, reorderedPad),
                () -> assertNotEquals(
                        new PadAttrs(List.of(), List.of(), v(0.0d)),
                        new PadAttrs(List.of(), List.of(), v(-0.0d))),
                () -> assertEquals(
                        "PadAttrs[before=[1, 0], after=[2, 3], "
                                + "constantValue=ScalarValue[dataType=FLOAT64, "
                                + "bits=0xBFF0000000000000]]",
                        pad.toString()),
                () -> assertEquals(tile, equalTile),
                () -> assertEquals(tile.hashCode(), equalTile.hashCode()),
                () -> assertNotEquals(tile, reorderedTile),
                () -> assertEquals("TileAttrs[repeats=[2, 3]]", tile.toString()));
    }

    @Test
    void composesOnlyTheDocumentedTypedPairsAndRetainsReferences() {
        PadAttrs padAttrs = new PadAttrs(List.of(1L), List.of(2L), v(-1.0d));
        TileAttrs tileAttrs = new TileAttrs(List.of(2L, 3L));
        Operation padded = new Operation(PadKind.PAD, padAttrs);
        Operation tiled = new Operation(TileKind.TILE, tileAttrs);

        assertAll(
                () -> assertSame(PadKind.PAD, padded.kind()),
                () -> assertSame(padAttrs, padded.attrs()),
                () -> assertSame(TileKind.TILE, tiled.kind()),
                () -> assertSame(tileAttrs, tiled.attrs()));
    }

    @Test
    void attributesContainOnlyTheirSpecifiedJavaBaseState() {
        List<String> padComponentTypes = Arrays.stream(PadAttrs.class.getRecordComponents())
                .map(component -> component.getGenericType().getTypeName())
                .toList();
        List<String> tileComponentTypes = Arrays.stream(TileAttrs.class.getRecordComponents())
                .map(component -> component.getGenericType().getTypeName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of(
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Long>",
                                "io.github.pho001.synaptik.model.datatype.ScalarValue"),
                        padComponentTypes),
                () -> assertEquals(
                        List.of("java.util.List<java.lang.Long>"), tileComponentTypes),
                () -> assertFalse(padComponentTypes.stream()
                        .anyMatch(PadTileSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(tileComponentTypes.stream()
                        .anyMatch(PadTileSemanticsTest::isForbiddenComponentType)));
    }

    private static <E extends Enum<E> & OperationKind> void assertEnumShape(
            Class<E> enumClass, E constant) {
        var constructors = enumClass.getDeclaredConstructors();
        var fields = enumClass.getDeclaredFields();
        var methods = enumClass.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        enumClass.getPackageName()),
                () -> assertTrue(Modifier.isPublic(enumClass.getModifiers())),
                () -> assertTrue(Modifier.isFinal(enumClass.getModifiers())),
                () -> assertTrue(enumClass.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class), Arrays.asList(enumClass.getInterfaces())),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(String.class, int.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !field.isEnumConstant())
                        .allMatch(field ->
                                field.isSynthetic() && Modifier.isStatic(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(
                        List.of("valueOf(java.lang.String):" + enumClass.getName(),
                                        "values():[L" + enumClass.getName() + ";")
                                .stream()
                                .sorted()
                                .toList(),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(PadTileSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, enumClass.getDeclaredClasses().length),
                () -> assertSame(enumClass, constant.getClass()));
    }

    private static void assertRecordType(Class<?> recordClass) {
        assertEquals(
                "io.github.pho001.synaptik.model.operation.layout", recordClass.getPackageName());
        assertTrue(Modifier.isPublic(recordClass.getModifiers()));
        assertTrue(Modifier.isFinal(recordClass.getModifiers()));
        assertTrue(recordClass.isRecord());
        assertEquals(List.of(OperationAttrs.class), Arrays.asList(recordClass.getInterfaces()));
    }

    private static List<String> declaredMethodSignatures(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .map(PadTileSemanticsTest::methodSignature)
                .sorted()
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
                || name.contains("Shape")
                || name.contains("DataType")
                || name.contains("layout.Layout")
                || name.contains("storage")
                || name.contains("graph")
                || name.contains("compiler")
                || name.contains("planning")
                || name.contains("prepare")
                || name.contains("runtime")
                || name.contains("backend")
                || name.contains("onnx")
                || name.contains("training");
    }

    private static ScalarValue v(double value) {
        return ScalarValue.float64(value);
    }
}
