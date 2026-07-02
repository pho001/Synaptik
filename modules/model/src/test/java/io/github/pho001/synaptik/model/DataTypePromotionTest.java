package io.github.pho001.synaptik.model;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class DataTypePromotionTest {
    private static final DataType[] FLOATING_TYPES = {
        DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64
    };

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
}
