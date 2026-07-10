package io.github.pho001.synaptik.model.datatype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import org.junit.jupiter.api.Test;

class DataTypePromotionTest {
    private static final DataType[] FLOATING_TYPES = {
        DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64
    };

    @Test
    void utilityHasExactlyTheTwoPublicPromotionMethods() throws ReflectiveOperationException {
        Method promoteFloating = DataTypePromotion.class.getDeclaredMethod(
                "promoteFloating", DataType.class, DataType.class);
        Method promoteNumeric = DataTypePromotion.class.getDeclaredMethod(
                "promoteNumeric", DataType.class, DataType.class);

        assertEquals(
                Set.of(promoteFloating, promoteNumeric),
                Set.copyOf(Arrays.asList(DataTypePromotion.class.getDeclaredMethods())).stream()
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .collect(java.util.stream.Collectors.toUnmodifiableSet()));
        assertTrue(Modifier.isFinal(DataTypePromotion.class.getModifiers()));
        assertEquals(0, DataTypePromotion.class.getDeclaredFields().length);
        assertEquals(DataType.class, promoteNumeric.getReturnType());
        assertTrue(Modifier.isPublic(promoteNumeric.getModifiers()));
        assertTrue(Modifier.isStatic(promoteNumeric.getModifiers()));
    }

    @Test
    void promotesEveryFloatingPairToTheWidestPrecision() {
        DataType[][] expected = {
            {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64},
            {DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT64},
            {DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64}
        };

        for (int left = 0; left < FLOATING_TYPES.length; left++) {
            for (int right = 0; right < FLOATING_TYPES.length; right++) {
                assertSame(
                        expected[left][right],
                        DataTypePromotion.promoteFloating(
                                FLOATING_TYPES[left], FLOATING_TYPES[right]));
            }
        }
    }

    @Test
    void rejectsNullOperands() {
        assertThrows(
                NullPointerException.class,
                () -> DataTypePromotion.promoteFloating(null, DataType.FLOAT32));
        assertThrows(
                NullPointerException.class,
                () -> DataTypePromotion.promoteFloating(DataType.FLOAT32, null));
    }

    @Test
    void rejectsIntegralAndBooleanOperandsOnEitherSide() {
        DataType[] unsupported = {DataType.INT32, DataType.INT64, DataType.BOOL};

        for (DataType dataType : unsupported) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> DataTypePromotion.promoteFloating(dataType, DataType.FLOAT32));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> DataTypePromotion.promoteFloating(DataType.FLOAT32, dataType));
        }
    }

    @Test
    void promotesEverySameCategoryNumericPairSymmetricallyAndIdempotently() {
        DataType[] integral = {DataType.INT32, DataType.INT64};

        for (DataType left : FLOATING_TYPES) {
            for (DataType right : FLOATING_TYPES) {
                assertAllPromotionLaws(left, right, DataTypePromotion.promoteFloating(left, right));
            }
        }
        for (DataType left : integral) {
            for (DataType right : integral) {
                DataType expected = left == DataType.INT64 || right == DataType.INT64
                        ? DataType.INT64
                        : DataType.INT32;
                assertAllPromotionLaws(left, right, expected);
            }
        }
    }

    @Test
    void numericPromotionRejectsNullBooleanAndMixedCategoryOperandsInExactOrder() {
        assertFailure(
                NullPointerException.class,
                "left",
                () -> DataTypePromotion.promoteNumeric(null, null));
        assertFailure(
                NullPointerException.class,
                "right",
                () -> DataTypePromotion.promoteNumeric(DataType.INT32, null));
        assertFailure(
                IllegalArgumentException.class,
                "left must be a numeric data type, but was BOOL",
                () -> DataTypePromotion.promoteNumeric(DataType.BOOL, DataType.BOOL));
        assertFailure(
                IllegalArgumentException.class,
                "right must be a numeric data type, but was BOOL",
                () -> DataTypePromotion.promoteNumeric(DataType.INT32, DataType.BOOL));
        assertFailure(
                IllegalArgumentException.class,
                "numeric data types must share a category, but were INT32 and BFLOAT16",
                () -> DataTypePromotion.promoteNumeric(DataType.INT32, DataType.BFLOAT16));
        assertFailure(
                IllegalArgumentException.class,
                "numeric data types must share a category, but were FLOAT64 and INT64",
                () -> DataTypePromotion.promoteNumeric(DataType.FLOAT64, DataType.INT64));
    }

    private static void assertAllPromotionLaws(
            DataType left, DataType right, DataType expected) {
        assertSame(expected, DataTypePromotion.promoteNumeric(left, right));
        assertSame(expected, DataTypePromotion.promoteNumeric(right, left));
        assertSame(left, DataTypePromotion.promoteNumeric(left, left));
        assertSame(right, DataTypePromotion.promoteNumeric(right, right));
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable executable) {
        T failure = assertThrows(type, executable);
        assertEquals(message, failure.getMessage());
    }
}
