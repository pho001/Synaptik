package io.github.pho001.synaptik.model.operation.elementwise.cast;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class CastValueConversionsTest {
    private static final List<DataType> MATRIX_ORDER = List.of(
            DataType.FLOAT64,
            DataType.FLOAT32,
            DataType.BFLOAT16,
            DataType.INT64,
            DataType.INT32,
            DataType.BOOL);

    @Test
    void exposesExactlyTheStatelessScalarConversionApi() {
        var constructors = CastValueConversions.class.getDeclaredConstructors();
        var publicOrProtectedMethods = Arrays.stream(CastValueConversions.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers())
                        || Modifier.isProtected(method.getModifiers()))
                .toList();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.cast",
                        CastValueConversions.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CastValueConversions.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CastValueConversions.class.getModifiers())),
                () -> assertFalse(CastValueConversions.class.isRecord()),
                () -> assertEquals(0, CastValueConversions.class.getInterfaces().length),
                () -> assertEquals(0, CastValueConversions.class.getDeclaredFields().length),
                () -> assertEquals(0, CastValueConversions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, publicOrProtectedMethods.size()),
                () -> assertEquals(
                        "convert(io.github.pho001.synaptik.model.datatype.ScalarValue,"
                                + "io.github.pho001.synaptik.model.datatype.DataType):"
                                + "io.github.pho001.synaptik.model.datatype.ScalarValue",
                        methodSignature(publicOrProtectedMethods.getFirst())),
                () -> assertTrue(Modifier.isStatic(
                        publicOrProtectedMethods.getFirst().getModifiers())));
    }

    @Test
    void checksSourceThenTargetWithExactMessages() {
        NullPointerException missingSource = assertThrows(
                NullPointerException.class,
                () -> CastValueConversions.convert(null, null));
        NullPointerException missingTarget = assertThrows(
                NullPointerException.class,
                () -> CastValueConversions.convert(ScalarValue.int32(1), null));

        assertAll(
                () -> assertEquals("source", missingSource.getMessage()),
                () -> assertEquals("targetDataType", missingTarget.getMessage()));
    }

    @Test
    void admitsAllThirtySixPairsWithIndependentRepresentativeResults() {
        ScalarValue[] sources = {
            ScalarValue.float64(-2.75d),
            ScalarValue.float32(-2.75f),
            ScalarValue.bfloat16Bits((short) 0xC030),
            ScalarValue.int64(-3L),
            ScalarValue.int32(-3),
            ScalarValue.bool(true)
        };
        ScalarValue[][] expected = {
            {
                ScalarValue.float64(-2.75d),
                ScalarValue.float32(-2.75f),
                ScalarValue.bfloat16Bits((short) 0xC030),
                ScalarValue.int64(-2L),
                ScalarValue.int32(-2),
                ScalarValue.bool(true)
            },
            {
                ScalarValue.float64(-2.75d),
                ScalarValue.float32(-2.75f),
                ScalarValue.bfloat16Bits((short) 0xC030),
                ScalarValue.int64(-2L),
                ScalarValue.int32(-2),
                ScalarValue.bool(true)
            },
            {
                ScalarValue.float64(-2.75d),
                ScalarValue.float32(-2.75f),
                ScalarValue.bfloat16Bits((short) 0xC030),
                ScalarValue.int64(-2L),
                ScalarValue.int32(-2),
                ScalarValue.bool(true)
            },
            {
                ScalarValue.float64(-3.0d),
                ScalarValue.float32(-3.0f),
                ScalarValue.bfloat16Bits((short) 0xC040),
                ScalarValue.int64(-3L),
                ScalarValue.int32(-3),
                ScalarValue.bool(true)
            },
            {
                ScalarValue.float64(-3.0d),
                ScalarValue.float32(-3.0f),
                ScalarValue.bfloat16Bits((short) 0xC040),
                ScalarValue.int64(-3L),
                ScalarValue.int32(-3),
                ScalarValue.bool(true)
            },
            {
                ScalarValue.float64(1.0d),
                ScalarValue.float32(1.0f),
                ScalarValue.bfloat16Bits((short) 0x3F80),
                ScalarValue.int64(1L),
                ScalarValue.int32(1),
                ScalarValue.bool(true)
            }
        };

        for (int sourceIndex = 0; sourceIndex < sources.length; sourceIndex++) {
            for (int targetIndex = 0; targetIndex < MATRIX_ORDER.size(); targetIndex++) {
                ScalarValue source = sources[sourceIndex];
                DataType target = MATRIX_ORDER.get(targetIndex);
                ScalarValue expectedResult = expected[sourceIndex][targetIndex];
                ScalarValue result = CastValueConversions.convert(source, target);

                assertAll(
                        () -> assertEquals(target, result.dataType()),
                        () -> assertEquals(expectedResult, result));
                if (source.dataType() == target) {
                    assertSame(source, result);
                } else {
                    assertNotSame(source, result);
                }
            }
        }
    }

    @Test
    void sameTypeConversionPreservesIdentityAndEverySelectedRawPattern() {
        List<ScalarValue> values = List.of(
                ScalarValue.float64(Double.longBitsToDouble(0xFFF0_0000_0000_0042L)),
                ScalarValue.float32(Float.intBitsToFloat(0x7FA1_2345)),
                ScalarValue.bfloat16Bits((short) 0xFF81),
                ScalarValue.int64(Long.MIN_VALUE),
                ScalarValue.int32(Integer.MIN_VALUE),
                ScalarValue.bool(false));

        for (ScalarValue value : values) {
            assertSame(value, CastValueConversions.convert(value, value.dataType()));
        }
    }

    @Test
    void roundsFloat64DirectlyToBFloat16AtTiesCarryOverflowAndUnderflow() {
        assertAll(
                () -> assertBFloat16(0x3F81, f64(0x3FF0_1000_0040_0000L)),
                () -> assertBFloat16(0xBF81, f64(0xBFF0_1000_0040_0000L)),
                () -> assertBFloat16(0x3F81, f64(0x3FF0_2FFF_FFC0_0000L)),
                () -> assertBFloat16(0xBF81, f64(0xBFF0_2FFF_FFC0_0000L)),
                () -> assertBFloat16(0x3F80, f64(0x3FF0_1000_0000_0000L)),
                () -> assertBFloat16(0x3F82, f64(0x3FF0_3000_0000_0000L)),
                () -> assertBFloat16(0x4000, f64(0x3FFF_F000_0000_0000L)),
                () -> assertBFloat16(0x7F7F, f64(0x47EF_EFFF_FFFF_FFFFL)),
                () -> assertBFloat16(0x7F80, f64(0x47EF_F000_0000_0000L)),
                () -> assertBFloat16(0xFF80, f64(0xC7EF_F000_0000_0000L)),
                () -> assertBFloat16(0x0000, f64(0x3790_0000_0000_0000L)),
                () -> assertBFloat16(0x0001, f64(0x3790_0000_0000_0001L)),
                () -> assertBFloat16(0x8000, f64(0xB790_0000_0000_0000L)),
                () -> assertBFloat16(0x8001, f64(0xB790_0000_0000_0001L)));
    }

    @Test
    void roundsLossyFloatingConversionsWithCanonicalNaNsAndTargetBoundaries() {
        assertAll(
                () -> assertFloat32Bits(0x3F80_0000, f64(0x3FF0_0000_1000_0000L)),
                () -> assertFloat32Bits(0x3F80_0002, f64(0x3FF0_0000_3000_0000L)),
                () -> assertFloat32Bits(0x0000_0000, f64(0x3690_0000_0000_0000L)),
                () -> assertFloat32Bits(0x0000_0001, f64(0x3690_0000_0000_0001L)),
                () -> assertFloat32Bits(0x7F7F_FFFF, f64(0x47EF_FFFF_EFFF_FFFFL)),
                () -> assertFloat32Bits(0x7F80_0000, f64(0x47EF_FFFF_F000_0000L)),
                () -> assertFloat32Bits(0x7FC0_0000, f64(0xFFF0_0000_0000_0001L)),
                () -> assertBFloat16(0x7FC0, f64(0xFFF0_0000_0000_0042L)),
                () -> assertBFloat16(0x0000, f32(0x0000_8000)),
                () -> assertBFloat16(0x0001, f32(0x0000_8001)),
                () -> assertBFloat16(0x7F7F, f32(0x7F7F_7FFF)),
                () -> assertBFloat16(0x7F80, f32(0x7F7F_8000)),
                () -> assertBFloat16(0x7FC0, f32(0xFFA1_2345)));
    }

    @Test
    void preservesFloatingSignsForZerosAndInfinities() {
        assertAll(
                () -> assertFloat32Bits(0x8000_0000, ScalarValue.float64(-0.0d)),
                () -> assertBFloat16(0x8000, ScalarValue.float64(-0.0d)),
                () -> assertFloat64Bits(0x8000_0000_0000_0000L,
                        ScalarValue.float32(-0.0f)),
                () -> assertBFloat16(0x8000, ScalarValue.float32(-0.0f)),
                () -> assertFloat32Bits(0x8000_0000,
                        ScalarValue.bfloat16Bits((short) 0x8000)),
                () -> assertFloat64Bits(0xFFF0_0000_0000_0000L,
                        ScalarValue.bfloat16Bits((short) 0xFF80)),
                () -> assertFloat32Bits(0x7F80_0000,
                        ScalarValue.bfloat16Bits((short) 0x7F80)),
                () -> assertBFloat16(0xFF80,
                        ScalarValue.float32(Float.NEGATIVE_INFINITY)));
    }

    @Test
    void widensFiniteValuesAndNaNsByExactLeftAlignedBitMappings() {
        ScalarValue bf16Signaling = ScalarValue.bfloat16Bits((short) 0x7F81);
        ScalarValue bf16QuietNegative = ScalarValue.bfloat16Bits((short) 0xFFC1);
        ScalarValue float32Signaling = f32(0x7FA1_2345);

        assertAll(
                () -> assertEquals(
                        0x7F81_0000,
                        Float.floatToRawIntBits(CastValueConversions
                                .convert(bf16Signaling, DataType.FLOAT32)
                                .float32Value())),
                () -> assertEquals(
                        0xFFF8_2000_0000_0000L,
                        Double.doubleToRawLongBits(CastValueConversions
                                .convert(bf16QuietNegative, DataType.FLOAT64)
                                .float64Value())),
                () -> assertEquals(
                        0x7FF4_2468_A000_0000L,
                        Double.doubleToRawLongBits(CastValueConversions
                                .convert(float32Signaling, DataType.FLOAT64)
                                .float64Value())),
                () -> assertFloat32Bits(0x0001_0000,
                        ScalarValue.bfloat16Bits((short) 0x0001)),
                () -> assertFloat64Bits(0x37A0_0000_0000_0000L,
                        ScalarValue.bfloat16Bits((short) 0x0001)),
                () -> assertFloat64Bits(0x36A0_0000_0000_0000L, f32(0x0000_0001)),
                () -> assertFloat64Bits(0x8000_0000_0000_0000L,
                        ScalarValue.bfloat16Bits((short) 0x8000)));
    }

    @Test
    void convertsIntegersDirectlyToFloatingFormatsAndAppliesWidthRules() {
        assertAll(
                () -> assertBFloat16(0x4E81, ScalarValue.int32(1_077_936_129)),
                () -> assertBFloat16(0xCE81, ScalarValue.int32(-1_077_936_129)),
                () -> assertBFloat16(0x4F01, ScalarValue.int64(2_155_872_257L)),
                () -> assertBFloat16(0xCF01, ScalarValue.int64(-2_155_872_257L)),
                () -> assertBFloat16(0x4380, ScalarValue.int32(257)),
                () -> assertBFloat16(0x4382, ScalarValue.int32(259)),
                () -> assertBFloat16(0x4400, ScalarValue.int32(511)),
                () -> assertFloat32Bits(0x4F00_0000, ScalarValue.int32(Integer.MAX_VALUE)),
                () -> assertFloat64Bits(0x43E0_0000_0000_0000L,
                        ScalarValue.int64(Long.MAX_VALUE)),
                () -> assertFloat64Bits(0xC3E0_0000_0000_0000L,
                        ScalarValue.int64(Long.MIN_VALUE)),
                () -> assertEquals(
                        Integer.MIN_VALUE,
                        CastValueConversions.convert(
                                        ScalarValue.int32(Integer.MIN_VALUE), DataType.INT64)
                                .int64Value()),
                () -> assertEquals(
                        Integer.MAX_VALUE,
                        CastValueConversions.convert(
                                        ScalarValue.int32(Integer.MAX_VALUE), DataType.INT64)
                                .int64Value()),
                () -> assertEquals(1, CastValueConversions.convert(
                                ScalarValue.int64(0x0000_0001_0000_0001L), DataType.INT32)
                        .int32Value()),
                () -> assertEquals(-1, CastValueConversions.convert(
                                ScalarValue.int64(0x0000_0000_FFFF_FFFFL), DataType.INT32)
                        .int32Value()),
                () -> assertEquals(0, CastValueConversions.convert(
                                ScalarValue.int64(Long.MIN_VALUE), DataType.INT32)
                        .int32Value()),
                () -> assertEquals(-1, CastValueConversions.convert(
                                ScalarValue.int64(Long.MAX_VALUE), DataType.INT32)
                        .int32Value()));
    }

    @Test
    void truncatesAndSaturatesFloatingValuesToSignedIntegers() {
        assertAll(
                () -> assertInt32(2_147_483_646, ScalarValue.float64(2_147_483_646.9d)),
                () -> assertInt32(Integer.MAX_VALUE, ScalarValue.float64(2_147_483_647.9d)),
                () -> assertInt32(Integer.MIN_VALUE, ScalarValue.float64(-2_147_483_648.9d)),
                () -> assertInt32(2_147_483_520, f32(0x4EFF_FFFF)),
                () -> assertInt32(Integer.MAX_VALUE, f32(0x4F00_0000)),
                () -> assertInt32(Integer.MIN_VALUE, f32(0xCF00_0000)),
                () -> assertInt64(9_223_372_036_854_774_784L,
                        f64(0x43DF_FFFF_FFFF_FFFFL)),
                () -> assertInt64(Long.MAX_VALUE, f64(0x43E0_0000_0000_0000L)),
                () -> assertInt64(Long.MIN_VALUE, f64(0xC3E0_0000_0000_0000L)),
                () -> assertInt64(0L, f64(0x7FF8_0000_0000_0042L)),
                () -> assertInt32(0, f32(0xFFA1_2345)),
                () -> assertInt64(Long.MAX_VALUE, ScalarValue.float64(Double.POSITIVE_INFINITY)),
                () -> assertInt64(Long.MIN_VALUE, ScalarValue.float64(Double.NEGATIVE_INFINITY)),
                () -> assertInt32(Integer.MAX_VALUE,
                        ScalarValue.bfloat16Bits((short) 0x7F80)),
                () -> assertInt32(Integer.MIN_VALUE,
                        ScalarValue.bfloat16Bits((short) 0xFF80)),
                () -> assertInt64(0L, ScalarValue.bfloat16Bits((short) 0x0001)),
                () -> assertInt64(0L, ScalarValue.bfloat16Bits((short) 0x007F)),
                () -> assertInt64(0L, f32(0x0000_0001)),
                () -> assertInt64(0L, f32(0x007F_FFFF)),
                () -> assertInt64(0L, f64(0x0000_0000_0000_0001L)),
                () -> assertInt64(0L, f64(0x000F_FFFF_FFFF_FFFFL)),
                () -> assertInt32(0, ScalarValue.float64(0.0d)),
                () -> assertInt32(0, ScalarValue.float64(-0.0d)),
                () -> assertInt32(42, ScalarValue.float64(42.0d)),
                () -> assertInt64(-2L, ScalarValue.float32(-2.75f)));
    }

    @Test
    void appliesCanonicalBooleanZeroOneAndNumericTruthiness() {
        assertAll(
                () -> assertNumericBoolean(false, 0x0000_0000_0000_0000L, 0x0000_0000, 0x0000),
                () -> assertNumericBoolean(true, 0x3FF0_0000_0000_0000L, 0x3F80_0000, 0x3F80),
                () -> assertFalse(toBoolean(ScalarValue.float64(0.0d))),
                () -> assertFalse(toBoolean(ScalarValue.float64(-0.0d))),
                () -> assertTrue(toBoolean(f64(0x0000_0000_0000_0001L))),
                () -> assertTrue(toBoolean(f64(0x8000_0000_0000_0001L))),
                () -> assertTrue(toBoolean(ScalarValue.float64(Double.POSITIVE_INFINITY))),
                () -> assertTrue(toBoolean(f64(0xFFF0_0000_0000_0001L))),
                () -> assertFalse(toBoolean(ScalarValue.float32(-0.0f))),
                () -> assertTrue(toBoolean(f32(0x0000_0001))),
                () -> assertTrue(toBoolean(f32(0xFFC0_0042))),
                () -> assertFalse(toBoolean(ScalarValue.bfloat16Bits((short) 0x8000))),
                () -> assertTrue(toBoolean(ScalarValue.bfloat16Bits((short) 0x8001))),
                () -> assertTrue(toBoolean(ScalarValue.bfloat16Bits((short) 0x7F81))),
                () -> assertFalse(toBoolean(ScalarValue.int64(0L))),
                () -> assertTrue(toBoolean(ScalarValue.int64(Long.MIN_VALUE))),
                () -> assertFalse(toBoolean(ScalarValue.int32(0))),
                () -> assertTrue(toBoolean(ScalarValue.int32(-1))));
    }

    private static ScalarValue f64(long bits) {
        return ScalarValue.float64(Double.longBitsToDouble(bits));
    }

    private static ScalarValue f32(int bits) {
        return ScalarValue.float32(Float.intBitsToFloat(bits));
    }

    private static void assertBFloat16(int expectedBits, ScalarValue source) {
        assertEquals(
                (short) expectedBits,
                CastValueConversions.convert(source, DataType.BFLOAT16).bfloat16Bits());
    }

    private static void assertFloat32Bits(int expectedBits, ScalarValue source) {
        assertEquals(
                expectedBits,
                Float.floatToRawIntBits(CastValueConversions
                        .convert(source, DataType.FLOAT32)
                        .float32Value()));
    }

    private static void assertFloat64Bits(long expectedBits, ScalarValue source) {
        assertEquals(
                expectedBits,
                Double.doubleToRawLongBits(CastValueConversions
                        .convert(source, DataType.FLOAT64)
                        .float64Value()));
    }

    private static void assertInt32(int expected, ScalarValue source) {
        assertEquals(
                expected,
                CastValueConversions.convert(source, DataType.INT32).int32Value());
    }

    private static void assertInt64(long expected, ScalarValue source) {
        assertEquals(
                expected,
                CastValueConversions.convert(source, DataType.INT64).int64Value());
    }

    private static boolean toBoolean(ScalarValue source) {
        return CastValueConversions.convert(source, DataType.BOOL).booleanValue();
    }

    private static void assertNumericBoolean(
            boolean source, long float64Bits, int float32Bits, int bfloat16Bits) {
        ScalarValue value = ScalarValue.bool(source);
        assertAll(
                () -> assertEquals(float64Bits, Double.doubleToRawLongBits(
                        CastValueConversions.convert(value, DataType.FLOAT64).float64Value())),
                () -> assertEquals(float32Bits, Float.floatToRawIntBits(
                        CastValueConversions.convert(value, DataType.FLOAT32).float32Value())),
                () -> assertEquals((short) bfloat16Bits,
                        CastValueConversions.convert(value, DataType.BFLOAT16).bfloat16Bits()),
                () -> assertEquals(source ? 1L : 0L,
                        CastValueConversions.convert(value, DataType.INT64).int64Value()),
                () -> assertEquals(source ? 1 : 0,
                        CastValueConversions.convert(value, DataType.INT32).int32Value()));
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName() + "(" + parameters + "):" + method.getReturnType().getName();
    }
}
