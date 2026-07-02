package io.github.pho001.synaptik.model;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DataTypeTest {
    @Test
    void definesExactlyTheInitialDataTypesInStableOrder() {
        assertArrayEquals(
                new DataType[] {
                    DataType.FLOAT64,
                    DataType.FLOAT32,
                    DataType.BFLOAT16,
                    DataType.INT32,
                    DataType.INT64,
                    DataType.BOOL
                },
                DataType.values());
    }

    @Test
    void definesExactlyTheRequiredDataTypeCategories() {
        assertArrayEquals(
                new DataTypeCategory[] {
                    DataTypeCategory.FLOATING,
                    DataTypeCategory.INTEGRAL,
                    DataTypeCategory.BOOLEAN
                },
                DataTypeCategory.values());
    }

    @Test
    void exposesExactMetadataForEveryDataType() {
        assertMetadata(DataType.FLOAT64, DataTypeCategory.FLOATING, 64, 8, true);
        assertMetadata(DataType.FLOAT32, DataTypeCategory.FLOATING, 32, 4, true);
        assertMetadata(DataType.BFLOAT16, DataTypeCategory.FLOATING, 16, 2, true);
        assertMetadata(DataType.INT32, DataTypeCategory.INTEGRAL, 32, 4, false);
        assertMetadata(DataType.INT64, DataTypeCategory.INTEGRAL, 64, 8, false);
        assertMetadata(DataType.BOOL, DataTypeCategory.BOOLEAN, 8, 1, false);
    }

    @Test
    void exposesMutuallyExclusiveCategoryPredicates() {
        for (DataType dataType : DataType.values()) {
            int matchingPredicates = (dataType.isFloating() ? 1 : 0)
                    + (dataType.isIntegral() ? 1 : 0)
                    + (dataType.isBoolean() ? 1 : 0);
            assertEquals(1, matchingPredicates, () -> dataType + " must have exactly one category");
            assertEquals(dataType.category() == DataTypeCategory.FLOATING, dataType.isFloating());
            assertEquals(dataType.category() == DataTypeCategory.INTEGRAL, dataType.isIntegral());
            assertEquals(dataType.category() == DataTypeCategory.BOOLEAN, dataType.isBoolean());
        }
    }

    @Test
    void limitsDifferentiabilityToFloatingDataTypes() {
        assertAll(
                () -> assertTrue(DataType.FLOAT64.isDifferentiable()),
                () -> assertTrue(DataType.FLOAT32.isDifferentiable()),
                () -> assertTrue(DataType.BFLOAT16.isDifferentiable()),
                () -> assertFalse(DataType.INT32.isDifferentiable()),
                () -> assertFalse(DataType.INT64.isDifferentiable()),
                () -> assertFalse(DataType.BOOL.isDifferentiable()));
    }

    @Test
    void usesFloat32AsTheDefaultFloatingDataType() {
        assertSame(DataType.FLOAT32, DataType.defaultFloating());
    }

    private static void assertMetadata(
            DataType dataType,
            DataTypeCategory expectedCategory,
            int expectedBitWidth,
            int expectedByteWidth,
            boolean expectedDifferentiable) {
        assertAll(
                dataType.name(),
                () -> assertSame(expectedCategory, dataType.category()),
                () -> assertEquals(expectedBitWidth, dataType.bitWidth()),
                () -> assertEquals(expectedByteWidth, dataType.byteWidth()),
                () -> assertEquals(expectedDifferentiable, dataType.isDifferentiable()));
    }
}
