package io.github.pho001.synaptik.model.datatype;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScalarValueTest {
    @Test
    void exposesOnlyTheExactFinalClassShape() {
        var fields = Arrays.stream(ScalarValue.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        var constructors = ScalarValue.class.getDeclaredConstructors();

        assertAll(
                () -> assertEquals("io.github.pho001.synaptik.model.datatype",
                        ScalarValue.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ScalarValue.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ScalarValue.class.getModifiers())),
                () -> assertFalse(ScalarValue.class.isRecord()),
                () -> assertEquals(List.of("dataType", "bits"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertEquals(List.of(DataType.class, long.class),
                        fields.stream().map(field -> field.getType()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(List.of(DataType.class, long.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertEquals(0, ScalarValue.class.getDeclaredClasses().length),
                () -> assertEquals(
                        List.of(
                                "bfloat16(float):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "bfloat16Bits():short",
                                "bfloat16Bits(short):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "bool(boolean):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "booleanValue():boolean",
                                "dataType():io.github.pho001.synaptik.model.datatype.DataType",
                                "equals(java.lang.Object):boolean",
                                "float32(float):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "float32Value():float",
                                "float64(double):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "float64Value():double",
                                "hashCode():int",
                                "int32(int):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "int32Value():int",
                                "int64(long):io.github.pho001.synaptik.model.datatype.ScalarValue",
                                "int64Value():long",
                                "toString():java.lang.String"),
                        Arrays.stream(ScalarValue.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(ScalarValueTest::methodSignature)
                                .sorted()
                                .toList()));
    }

    @Test
    void preservesEveryCurrentRepresentationExactly() {
        double float64NaN = Double.longBitsToDouble(0xFFF8_0000_0000_0042L);
        float float32NaN = Float.intBitsToFloat(0xFFC0_0042);
        long large = 9_007_199_254_740_993L;

        assertAll(
                () -> assertEquals(0x8000_0000_0000_0000L,
                        Double.doubleToRawLongBits(ScalarValue.float64(-0.0d).float64Value())),
                () -> assertEquals(0xFFF8_0000_0000_0042L,
                        Double.doubleToRawLongBits(ScalarValue.float64(float64NaN).float64Value())),
                () -> assertEquals(0x8000_0000,
                        Float.floatToRawIntBits(ScalarValue.float32(-0.0f).float32Value())),
                () -> assertEquals(0xFFC0_0042,
                        Float.floatToRawIntBits(ScalarValue.float32(float32NaN).float32Value())),
                () -> assertEquals((short) 0xFFC1,
                        ScalarValue.bfloat16Bits((short) 0xFFC1).bfloat16Bits()),
                () -> assertEquals(BFloat16Bits.fromFloat(1.00390625f),
                        ScalarValue.bfloat16(1.00390625f).bfloat16Bits()),
                () -> assertEquals(Integer.MIN_VALUE,
                        ScalarValue.int32(Integer.MIN_VALUE).int32Value()),
                () -> assertEquals(large, ScalarValue.int64(large).int64Value()),
                () -> assertFalse(ScalarValue.bool(false).booleanValue()),
                () -> assertTrue(ScalarValue.bool(true).booleanValue()));
    }

    @Test
    void strictInspectorsRejectEveryMismatchedTypeWithExactMessage() {
        ScalarValue value = ScalarValue.int64(1L);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class, value::float64Value);

        assertEquals("scalar value has data type INT64, not FLOAT64", failure.getMessage());
    }

    @Test
    void usesExactTypedBitsForEqualityHashingAndDiagnosticText() {
        ScalarValue nanOne = ScalarValue.float32(Float.intBitsToFloat(0x7FC0_0001));
        ScalarValue nanTwo = ScalarValue.float32(Float.intBitsToFloat(0x7FC0_0002));

        assertAll(
                () -> assertEquals(ScalarValue.int64(Long.MIN_VALUE),
                        ScalarValue.int64(Long.MIN_VALUE)),
                () -> assertEquals(ScalarValue.int64(Long.MIN_VALUE).hashCode(),
                        ScalarValue.int64(Long.MIN_VALUE).hashCode()),
                () -> assertNotEquals(ScalarValue.float64(1.0d), ScalarValue.float32(1.0f)),
                () -> assertNotEquals(ScalarValue.float64(0.0d), ScalarValue.float64(-0.0d)),
                () -> assertNotEquals(ScalarValue.float32(0.0f), ScalarValue.float32(-0.0f)),
                () -> assertNotEquals(nanOne, nanTwo),
                () -> assertNotEquals(ScalarValue.bfloat16Bits((short) 0x7FC1),
                        ScalarValue.bfloat16Bits((short) 0x7FC2)),
                () -> assertEquals(
                        "ScalarValue[dataType=FLOAT64, bits=0x8000000000000000]",
                        ScalarValue.float64(-0.0d).toString()),
                () -> assertEquals("ScalarValue[dataType=FLOAT32, bits=0x80000000]",
                        ScalarValue.float32(-0.0f).toString()),
                () -> assertEquals("ScalarValue[dataType=BFLOAT16, bits=0xFFC1]",
                        ScalarValue.bfloat16Bits((short) 0xFFC1).toString()),
                () -> assertEquals("ScalarValue[dataType=INT32, bits=0xFFFFFFFF]",
                        ScalarValue.int32(-1).toString()),
                () -> assertEquals("ScalarValue[dataType=INT64, bits=0xFFFFFFFFFFFFFFFF]",
                        ScalarValue.int64(-1L).toString()),
                () -> assertEquals("ScalarValue[dataType=BOOL, bits=0x01]",
                        ScalarValue.bool(true).toString()));
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName() + "(" + parameters + "):" + method.getReturnType().getName();
    }
}
