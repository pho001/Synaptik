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
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.datatype.DataType;
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
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(ScalarElementwiseKind.class);
    }

    @Test
    void exposesOnlyTheExactRecordShapes() {
        assertAll(
                () -> assertRecordShape(
                        ScalarValueAttrs.class,
                        List.of("value"),
                        List.of(ScalarValue.class),
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String",
                                "value():io.github.pho001.synaptik.model.datatype.ScalarValue")),
                () -> assertRecordShape(
                        ClampRangeAttrs.class,
                        List.of("minValue", "maxValue"),
                        List.of(ScalarValue.class, ScalarValue.class),
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "maxValue():io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "minValue():io.github.pho001.synaptik.model.datatype.ScalarValue",
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
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                BinaryArithmeticKind.MUL,
                                new ScalarValueAttrs(v(2.0)))),
                () -> assertEquals(
                        ScalarElementwiseKind.MUL,
                        new Operation(
                                        ScalarElementwiseKind.MUL,
                                        new ScalarValueAttrs(v(2.0)))
                                .kind()));
    }

    @Test
    void composesEveryKindWithItsDocumentedExactAttributesReference() {
        ScalarValueAttrs multiplier = new ScalarValueAttrs(v(0.5));
        ScalarValueAttrs exponent = new ScalarValueAttrs(v(2.0));
        ClampRangeAttrs range = new ClampRangeAttrs(v(-1.0), v(1.0));
        ScalarValueAttrs minimum = new ScalarValueAttrs(v(-3.0));
        ScalarValueAttrs maximum = new ScalarValueAttrs(v(3.0));

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
            ScalarValueAttrs attrs = new ScalarValueAttrs(
                    ScalarValue.float64(Double.longBitsToDouble(bits)));

            assertEquals(bits, Double.doubleToRawLongBits(attrs.value().float64Value()));
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
                    () -> new ClampRangeAttrs(v(values[0]), v(values[1])));

            assertEquals(
                    "minValue must be less than or equal to maxValue",
                    failure.getMessage());
        }
    }

    @Test
    void validatesClampTypesNumericDomainAndExactPrimitiveOrdering() {
        assertNullFailure(() -> new ScalarValueAttrs(null), "value");
        assertNullFailure(() -> new ClampRangeAttrs(null, null), "minValue");
        assertNullFailure(() -> new ClampRangeAttrs(v(0.0), null), "maxValue");
        assertIllegalFailure(
                () -> new ClampRangeAttrs(ScalarValue.float32(0.0f), v(1.0)),
                "minValue and maxValue must have the same data type: FLOAT32 != FLOAT64");
        assertIllegalFailure(
                () -> new ClampRangeAttrs(ScalarValue.bool(false), ScalarValue.bool(true)),
                "clamp bounds must be numeric, but were BOOL");
        assertIllegalFailure(
                () -> new ClampRangeAttrs(ScalarValue.int64(9_007_199_254_740_993L),
                        ScalarValue.int64(9_007_199_254_740_992L)),
                "minValue must be less than or equal to maxValue");

        assertAll(
                () -> assertSame(DataType.FLOAT32,
                        new ClampRangeAttrs(
                                ScalarValue.float32(Float.NaN), ScalarValue.float32(-1.0f))
                                .minValue().dataType()),
                () -> assertEquals((short) 0x7FC1,
                        new ClampRangeAttrs(
                                ScalarValue.bfloat16Bits((short) 0x7FC1),
                                ScalarValue.bfloat16Bits((short) 0x3F80))
                                .minValue().bfloat16Bits()),
                () -> assertEquals(Integer.MIN_VALUE,
                        new ClampRangeAttrs(
                                ScalarValue.int32(Integer.MIN_VALUE), ScalarValue.int32(0))
                                .minValue().int32Value()),
                () -> assertEquals(9_007_199_254_740_993L,
                        new ClampRangeAttrs(
                                ScalarValue.int64(9_007_199_254_740_993L),
                                ScalarValue.int64(Long.MAX_VALUE))
                                .minValue().int64Value()));
    }

    @Test
    void usesGeneratedRecordEqualityHashingAndDiagnosticText() {
        ScalarValueAttrs scalar = new ScalarValueAttrs(v(2.5));
        ScalarValueAttrs equalScalar = new ScalarValueAttrs(v(2.5));
        ScalarValueAttrs differentScalar = new ScalarValueAttrs(v(-2.5));
        ClampRangeAttrs range = new ClampRangeAttrs(v(-1.0), v(1.0));
        ClampRangeAttrs equalRange = new ClampRangeAttrs(v(-1.0), v(1.0));
        ClampRangeAttrs differentRange = new ClampRangeAttrs(v(-1.0), v(2.0));

        assertAll(
                () -> assertEquals(scalar, equalScalar),
                () -> assertEquals(scalar.hashCode(), equalScalar.hashCode()),
                () -> assertNotEquals(scalar, differentScalar),
                () -> assertNotEquals(new ScalarValueAttrs(v(0.0)), new ScalarValueAttrs(v(-0.0))),
                () -> assertNotEquals(
                        new ScalarValueAttrs(ScalarValue.float64(
                                Double.longBitsToDouble(0x7ff8_0000_0000_0001L))),
                        new ScalarValueAttrs(ScalarValue.float64(
                                Double.longBitsToDouble(0x7ff8_0000_0000_0042L)))),
                () -> assertEquals(range, equalRange),
                () -> assertEquals(range.hashCode(), equalRange.hashCode()),
                () -> assertNotEquals(range, differentRange),
                () -> assertEquals(
                        "ScalarValueAttrs[value=ScalarValue[dataType=FLOAT64, "
                                + "bits=0x4004000000000000]]",
                        scalar.toString()),
                () -> assertEquals(
                        "ClampRangeAttrs[minValue=ScalarValue[dataType=FLOAT64, "
                                + "bits=0xBFF0000000000000], maxValue=ScalarValue[dataType=FLOAT64, "
                                + "bits=0x3FF0000000000000]]",
                        range.toString()));
    }

    private static void assertComposition(ScalarElementwiseKind kind, OperationAttrs attrs) {
        Operation operation = new Operation(kind, attrs);

        assertAll(
                () -> assertSame(kind, operation.kind()),
                () -> assertSame(attrs, operation.attrs()));
    }

    private static void assertClampBits(double minValue, double maxValue) {
        ClampRangeAttrs attrs = new ClampRangeAttrs(v(minValue), v(maxValue));

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(minValue),
                        Double.doubleToRawLongBits(attrs.minValue().float64Value())),
                () -> assertEquals(
                        Double.doubleToRawLongBits(maxValue),
                        Double.doubleToRawLongBits(attrs.maxValue().float64Value())));
    }

    private static ScalarValue v(double value) {
        return ScalarValue.float64(value);
    }

    private static void assertNullFailure(
            org.junit.jupiter.api.function.Executable executable, String message) {
        assertEquals(message, assertThrows(NullPointerException.class, executable).getMessage());
    }

    private static void assertIllegalFailure(
            org.junit.jupiter.api.function.Executable executable, String message) {
        assertEquals(message,
                assertThrows(IllegalArgumentException.class, executable).getMessage());
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
