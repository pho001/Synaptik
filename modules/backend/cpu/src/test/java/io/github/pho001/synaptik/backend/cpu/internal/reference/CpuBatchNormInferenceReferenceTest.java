package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuBatchNormInferenceReferenceTest {
    @Test void preservesDirectVarianceFormulaOrderAndExceptionalClasses() {
        double[][] inputs = {{1, 2, -0.0, Double.POSITIVE_INFINITY}, {2, 0}, {.5, -1},
                {1, 0}, {4, -1}};
        long[][] extents = {{2, 2}, {2}, {2}, {2}, {2}};
        long[][] strides = {{2, 1}, {1}, {1}, {1}, {1}};
        double[] result = CpuBatchNormInferenceReferenceKernel.evaluate(
                java.util.Collections.nCopies(5, DataType.FLOAT64), DataType.FLOAT64, 1e-5,
                inputs, extents, new long[5], strides, 1);
        assertAll(() -> assertEquals(((1 - 1) / Math.sqrt(4 + 1e-5)) * 2 + .5,
                                result[0], 0),
                () -> assertEquals(((2 - 0) / Math.sqrt(-1 + 1e-5)) * 0 - 1,
                        result[1]),
                () -> assertEquals(((-0.0 - 1) / Math.sqrt(4 + 1e-5)) * 2 + .5,
                                result[2], 0),
                () -> assertTrue(Double.isNaN(result[3])));
    }

    @Test void roundsEveryFloatOperationAndEncodesBfloatOnlyAtFinalResult() {
        double[][] inputs = {{1.00390625}, {1.0078125}, {.00390625}, {.0078125}, {.015625}};
        long[][] extents = {{1, 1}, {1}, {1}, {1}, {1}};
        long[][] strides = {{1, 1}, {1}, {1}, {1}, {1}};
        double[] result = CpuBatchNormInferenceReferenceKernel.evaluate(
                java.util.Collections.nCopies(5, DataType.BFLOAT16), DataType.BFLOAT16,
                Float.intBitsToFloat(0x3728 << 16), inputs, extents, new long[5], strides, 1);
        assertEquals(0, Float.floatToRawIntBits((float) result[0]) & 0xffff);
    }
}
