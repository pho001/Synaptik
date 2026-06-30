package runtime.device.buffer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorBufferLayoutClassifierTest {
    @Test
    void denseContiguousLayoutClassifiesAsDenseContiguous() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );

        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, layout.layoutClass());
        assertEquals(24, layout.logicalByteLength());
    }

    @Test
    void zeroOffsetPaddedViewClassifiesAsZeroOffsetView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT64,
                new int[]{2, 2},
                new int[]{4, 2},
                0,
                4
        );

        assertEquals(AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW, layout.layoutClass());
    }

    @Test
    void nonZeroOffsetDenseViewClassifiesAsNonZeroOffsetView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                2,
                6
        );

        assertEquals(AcceleratorBufferLayoutClass.NON_ZERO_OFFSET_VIEW, layout.layoutClass());
        assertEquals(2, layout.storageOffset());
    }

    @Test
    void permutedLayoutClassifiesAsPermutedOrStridedView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{1, 2},
                0,
                6
        );

        assertEquals(AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW, layout.layoutClass());
    }

    @Test
    void broadcastZeroStrideClassifiesAsBroadcastZeroStrideView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.BFLOAT16,
                new int[]{2, 3},
                new int[]{0, 1},
                0,
                6
        );

        assertEquals(AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW, layout.layoutClass());
        assertEquals(12, layout.logicalByteLength());
    }

    @Test
    void unsupportedNegativeStrideClassifiesAsUnsupported() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, -1},
                0,
                6
        );

        assertEquals(AcceleratorBufferLayoutClass.UNSUPPORTED, layout.layoutClass());
    }

    @Test
    void classifiesMismatchedLogicalElementCountAsUnsupported() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayoutClassifier.describe(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                5
        );

        assertEquals(AcceleratorBufferLayoutClass.UNSUPPORTED, layout.layoutClass());
    }

    @ParameterizedTest
    @CsvSource({
            "FLOAT32,4",
            "FLOAT64,8",
            "BFLOAT16,2",
            "INT32,4",
            "BOOL,1"
    })
    void computesLogicalByteLengthForSupportedDtypes(DataType dataType, int bytesPerElement) {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                dataType,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );

        assertEquals(6L * bytesPerElement, layout.logicalByteLength());
    }

    @Test
    void layoutAccessorsReturnDefensiveCopies() {
        int[] shape = {2, 3};
        int[] strides = {3, 1};
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(DataType.FLOAT32, shape, strides, 0, 6);

        shape[0] = 99;
        strides[0] = 99;

        assertArrayEquals(new int[]{2, 3}, layout.shape());
        assertArrayEquals(new int[]{3, 1}, layout.strides());

        int[] returnedShape = layout.shape();
        int[] returnedStrides = layout.strides();
        returnedShape[0] = 77;
        returnedStrides[0] = 77;

        assertArrayEquals(new int[]{2, 3}, layout.shape());
        assertArrayEquals(new int[]{3, 1}, layout.strides());
    }

    @Test
    void rejectsInvalidLayoutInputs() {
        assertThrows(NullPointerException.class,
                () -> AcceleratorBufferLayout.of(null, new int[]{2}, new int[]{1}, 0, 2));
        assertThrows(NullPointerException.class,
                () -> AcceleratorBufferLayout.of(DataType.FLOAT32, null, new int[]{1}, 0, 2));
        assertThrows(NullPointerException.class,
                () -> AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2}, null, 0, 2));
        assertThrows(IllegalArgumentException.class,
                () -> AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2}, new int[]{1, 1}, 0, 2));
        assertThrows(IllegalArgumentException.class,
                () -> AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2}, new int[]{1}, -1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> new AcceleratorBufferLayout(
                        DataType.FLOAT32,
                        new int[]{2},
                        new int[]{1},
                        0,
                        -1,
                        4,
                        AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                ));
        assertThrows(IllegalArgumentException.class,
                () -> new AcceleratorBufferLayout(
                        DataType.FLOAT32,
                        new int[]{2},
                        new int[]{1},
                        0,
                        2,
                        -1,
                        AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                ));
        IllegalArgumentException mismatch = assertThrows(IllegalArgumentException.class,
                () -> new AcceleratorBufferLayout(
                        DataType.FLOAT32,
                        new int[]{2},
                        new int[]{1},
                        0,
                        2,
                        4,
                        AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                ));
        assertTrue(mismatch.getMessage().contains("does not match dtype/element count byte length"));
    }

    @Test
    void rejectsLogicalByteLengthOverflow() {
        assertThrows(ArithmeticException.class,
                () -> AcceleratorBufferLayout.of(DataType.FLOAT64, new int[]{1}, new int[]{1}, 0, Long.MAX_VALUE));
    }

    @Test
    void createsLayoutFromTensorWithoutChangingPublicTensorApi() {
        Tensor tensor = new Tensor(new int[]{2, 3}, new int[]{4, 2}, 1, List.of(), null, "view", DataType.INT32);

        AcceleratorBufferLayout layout = AcceleratorBufferLayout.fromTensor(tensor);

        assertEquals(DataType.INT32, layout.dataType());
        assertArrayEquals(new int[]{2, 3}, layout.shape());
        assertArrayEquals(new int[]{4, 2}, layout.strides());
        assertEquals(1, layout.storageOffset());
        assertEquals(6, layout.logicalElementCount());
        assertEquals(24, layout.logicalByteLength());
        assertEquals(AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW, layout.layoutClass());
    }
}
