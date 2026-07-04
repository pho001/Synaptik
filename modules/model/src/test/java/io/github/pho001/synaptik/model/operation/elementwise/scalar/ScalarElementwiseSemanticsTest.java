package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScalarElementwiseSemanticsTest {
    @Test
    void declaresExactlyTheRequiredKindVocabularyInOrder() {
        ScalarElementwiseKind[] values = ScalarElementwiseKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new ScalarElementwiseKind[] {
                            ScalarElementwiseKind.MUL,
                            ScalarElementwiseKind.POW,
                            ScalarElementwiseKind.CLAMP,
                            ScalarElementwiseKind.CLAMP_MIN,
                            ScalarElementwiseKind.CLAMP_MAX
                        },
                        values),
                () -> assertEquals(
                        List.of("MUL", "POW", "CLAMP", "CLAMP_MIN", "CLAMP_MAX"),
                        Arrays.stream(values).map(ScalarElementwiseKind::name).toList()),
                () -> assertInstanceOf(OperationKind.class, ScalarElementwiseKind.MUL),
                () -> assertSame(
                        ScalarElementwiseKind.CLAMP,
                        ScalarElementwiseKind.valueOf("CLAMP")));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = ScalarElementwiseKind.class.getDeclaredConstructors();
        var instanceFields =
                Arrays.stream(ScalarElementwiseKind.class.getDeclaredFields())
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .toList();
        var instanceMethods =
                Arrays.stream(ScalarElementwiseKind.class.getDeclaredMethods())
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .toList();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.scalar",
                        ScalarElementwiseKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ScalarElementwiseKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ScalarElementwiseKind.class.getModifiers())),
                () -> assertTrue(ScalarElementwiseKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(ScalarElementwiseKind.class.getInterfaces())),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(String.class, int.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertTrue(instanceFields.isEmpty()),
                () -> assertTrue(instanceMethods.isEmpty()),
                () -> assertEquals(
                        List.of(
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;"),
                        Arrays.stream(ScalarElementwiseKind.class.getDeclaredMethods())
                                .filter(method -> !method.isSynthetic())
                                .map(ScalarElementwiseSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, ScalarElementwiseKind.class.getDeclaredClasses().length),
                () -> assertTrue(Arrays.stream(ScalarElementwiseKind.values())
                        .allMatch(value -> value.getClass() == ScalarElementwiseKind.class)));
    }

    @Test
    void exposesOnlyTheExactRecordShapes() {
        assertAll(
                () -> assertRecordShape(
                        ScalarValueAttrs.class,
                        List.of("value"),
                        List.of(double.class),
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String",
                                "value():double")),
                () -> assertRecordShape(
                        ClampRangeAttrs.class,
                        List.of("minValue", "maxValue"),
                        List.of(double.class, double.class),
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "maxValue():double",
                                "minValue():double",
                                "toString():java.lang.String")));
    }

    @Test
    void keepsScalarKindsDistinctFromEquallyNamedBinaryKinds() {
        assertAll(
                () -> assertEquals(
                        BinaryArithmeticKind.MUL.name(), ScalarElementwiseKind.MUL.name()),
                () -> assertEquals(
                        BinaryArithmeticKind.POW.name(), ScalarElementwiseKind.POW.name()),
                () -> assertNotEquals(BinaryArithmeticKind.MUL, ScalarElementwiseKind.MUL),
                () -> assertNotEquals(BinaryArithmeticKind.POW, ScalarElementwiseKind.POW),
                () -> assertNotEquals(
                        new Operation(
                                BinaryArithmeticKind.MUL,
                                new ScalarValueAttrs(2.0)),
                        new Operation(
                                ScalarElementwiseKind.MUL,
                                new ScalarValueAttrs(2.0))));
    }

    @Test
    void composesEveryKindWithItsDocumentedExactAttributesReference() {
        ScalarValueAttrs multiplier = new ScalarValueAttrs(0.5);
        ScalarValueAttrs exponent = new ScalarValueAttrs(2.0);
        ClampRangeAttrs range = new ClampRangeAttrs(-1.0, 1.0);
        ScalarValueAttrs minimum = new ScalarValueAttrs(-3.0);
        ScalarValueAttrs maximum = new ScalarValueAttrs(3.0);

        assertComposition(ScalarElementwiseKind.MUL, multiplier);
        assertComposition(ScalarElementwiseKind.POW, exponent);
        assertComposition(ScalarElementwiseKind.CLAMP, range);
        assertComposition(ScalarElementwiseKind.CLAMP_MIN, minimum);
        assertComposition(ScalarElementwiseKind.CLAMP_MAX, maximum);
    }

    @Test
    void retainsEveryScalarValueBitPatternWithoutValidation() {
        long[] bitPatterns = {
            Double.doubleToRawLongBits(1.25),
            Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
            Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
            Double.doubleToRawLongBits(0.0),
            Double.doubleToRawLongBits(-0.0),
            0x7ff8_0000_0000_0001L,
            0x7ff8_0000_0000_0042L,
            0xfff8_0000_0000_0001L
        };

        for (long bits : bitPatterns) {
            ScalarValueAttrs attrs = new ScalarValueAttrs(Double.longBitsToDouble(bits));

            assertEquals(bits, Double.doubleToRawLongBits(attrs.value()));
        }
    }

    @Test
    void acceptsAndRetainsEveryNonInvertedClampEdgeCase() {
        assertClampBits(-2.0, 3.0);
        assertClampBits(4.0, 4.0);
        assertClampBits(-0.0, 0.0);
        assertClampBits(0.0, -0.0);
        assertClampBits(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        assertClampBits(Double.NEGATIVE_INFINITY, 0.0);
        assertClampBits(0.0, Double.POSITIVE_INFINITY);
        assertClampBits(Double.longBitsToDouble(0x7ff8_0000_0000_0001L), 1.0);
        assertClampBits(-1.0, Double.longBitsToDouble(0x7ff8_0000_0000_0042L));
        assertClampBits(
                Double.longBitsToDouble(0xfff8_0000_0000_0001L),
                Double.longBitsToDouble(0x7ff8_0000_0000_0042L));
    }

    @Test
    void rejectsOnlyStrictlyInvertedClampRangesWithTheExactMessage() {
        double[][] inverted = {
            {2.0, 1.0},
            {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY},
            {Double.POSITIVE_INFINITY, 0.0},
            {0.0, Double.NEGATIVE_INFINITY}
        };

        for (double[] values : inverted) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new ClampRangeAttrs(values[0], values[1]));

            assertEquals(
                    "minValue must be less than or equal to maxValue",
                    failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordEqualityHashingAndDiagnosticText() {
        ScalarValueAttrs scalar = new ScalarValueAttrs(2.5);
        ScalarValueAttrs equalScalar = new ScalarValueAttrs(2.5);
        ScalarValueAttrs differentScalar = new ScalarValueAttrs(-2.5);
        ClampRangeAttrs range = new ClampRangeAttrs(-1.0, 1.0);
        ClampRangeAttrs equalRange = new ClampRangeAttrs(-1.0, 1.0);
        ClampRangeAttrs differentRange = new ClampRangeAttrs(-1.0, 2.0);

        assertAll(
                () -> assertEquals(scalar, equalScalar),
                () -> assertEquals(scalar.hashCode(), equalScalar.hashCode()),
                () -> assertNotEquals(scalar, differentScalar),
                () -> assertNotEquals(new ScalarValueAttrs(0.0), new ScalarValueAttrs(-0.0)),
                () -> assertEquals(
                        new ScalarValueAttrs(Double.longBitsToDouble(0x7ff8_0000_0000_0001L)),
                        new ScalarValueAttrs(Double.longBitsToDouble(0x7ff8_0000_0000_0042L))),
                () -> assertEquals(range, equalRange),
                () -> assertEquals(range.hashCode(), equalRange.hashCode()),
                () -> assertNotEquals(range, differentRange),
                () -> assertEquals("ScalarValueAttrs[value=2.5]", scalar.toString()),
                () -> assertEquals(
                        "ClampRangeAttrs[minValue=-1.0, maxValue=1.0]", range.toString()));
    }

    private static void assertComposition(ScalarElementwiseKind kind, OperationAttrs attrs) {
        Operation operation = new Operation(kind, attrs);

        assertAll(
                () -> assertSame(kind, operation.kind()),
                () -> assertSame(attrs, operation.attrs()));
    }

    private static void assertClampBits(double minValue, double maxValue) {
        ClampRangeAttrs attrs = new ClampRangeAttrs(minValue, maxValue);

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(minValue),
                        Double.doubleToRawLongBits(attrs.minValue())),
                () -> assertEquals(
                        Double.doubleToRawLongBits(maxValue),
                        Double.doubleToRawLongBits(attrs.maxValue())));
    }

    private static void assertRecordShape(
            Class<?> type,
            List<String> componentNames,
            List<Class<?>> componentTypes,
            List<String> methodSignatures) {
        var components = type.getRecordComponents();
        var constructors = type.getDeclaredConstructors();
        var constructor = constructors[0];
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.scalar",
                        type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class), Arrays.asList(type.getInterfaces())),
                () -> assertEquals(componentNames, Arrays.stream(components)
                        .map(component -> component.getName())
                        .toList()),
                () -> assertEquals(componentTypes, Arrays.stream(components)
                        .map(component -> component.getType())
                        .toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(componentTypes, Arrays.asList(constructor.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertEquals(componentNames, instanceFields.stream()
                        .map(field -> field.getName())
                        .toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        methodSignatures,
                        Arrays.stream(type.getDeclaredMethods())
                                .map(ScalarElementwiseSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
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
}
