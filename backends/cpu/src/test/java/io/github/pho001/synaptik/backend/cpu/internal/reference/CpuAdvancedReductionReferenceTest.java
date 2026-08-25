package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuAdvancedReductionReferenceTest {
    @Test void independentlyCoversFiniteSpecialEmptyPointAndCorrectedStatistics() {
        assertAll(
                () -> assertEquals(3.0, one(CpuAdvancedReductionIr.Kind.L1_NORM,
                        new double[] {-1, 2}, new long[] {2}, new int[] {0}, 0)),
                () -> assertEquals(5.0, one(CpuAdvancedReductionIr.Kind.L2_NORM,
                        new double[] {3, 4}, new long[] {2}, new int[] {0}, 0)),
                () -> assertEquals(2.0, one(CpuAdvancedReductionIr.Kind.VARIANCE,
                        new double[] {1, 3}, new long[] {2}, new int[] {0}, 1)),
                () -> assertEquals(Double.NEGATIVE_INFINITY, one(
                        CpuAdvancedReductionIr.Kind.LOG_SUM_EXP, new double[0],
                        new long[] {0}, new int[] {0}, 0)),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(one(
                        CpuAdvancedReductionIr.Kind.LOG_SUM_EXP, new double[] {-0.0},
                        new long[] {1}, new int[] {0}, 0))),
                () -> assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(one(
                        CpuAdvancedReductionIr.Kind.L1_NORM,
                        new double[] {Double.POSITIVE_INFINITY, Double.NaN},
                        new long[] {2}, new int[] {0}, 0))));
    }

    private static double one(CpuAdvancedReductionIr.Kind kind, double[] input,
            long[] extents, int[] axes, long correction) {
        return CpuAdvancedReductionReferenceKernel.evaluate(kind, DataType.FLOAT64, input,
                extents, 0, new long[] {1}, axes, false, correction)[0];
    }
}
