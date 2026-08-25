package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuTrailingNormalizationReferenceTest {
    @Test void independentlyEvaluatesLayerConstantsAndScaledRms() {
        double[] constant = CpuTrailingNormalizationReferenceKernel.evaluate(
                CpuTrailingNormalizationIr.Kind.LAYER, CpuTrailingNormalizationIr.Form.LAYER,
                List.of(DataType.FLOAT64), DataType.FLOAT64, 1e-5,
                new double[][] {{-0.0, 0.0}}, new long[][] {{1, 2}}, new long[] {0},
                new long[][] {{2, 1}}, 1);
        double[] rms = CpuTrailingNormalizationReferenceKernel.evaluate(
                CpuTrailingNormalizationIr.Kind.RMS, CpuTrailingNormalizationIr.Form.RMS_SCALED,
                List.of(DataType.FLOAT64, DataType.FLOAT64), DataType.FLOAT64, 1e-5,
                new double[][] {{1, 2, 3}, {2, -1, .5}}, new long[][] {{1, 3}, {3}},
                new long[] {0, 0}, new long[][] {{3, 1}, {1}}, 1);
        double root = StrictMath.sqrt(14.0 / 3.0 + 1e-5);
        assertAll(() -> assertEquals(0L, Double.doubleToRawLongBits(constant[0])),
                () -> assertEquals(0L, Double.doubleToRawLongBits(constant[1])),
                () -> assertArrayEquals(new double[] {2 / root, -2 / root, 1.5 / root}, rms,
                        2e-15));
    }
}
