package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class CpuScalarPowerAnalysisTest {
    @ParameterizedTest
    @MethodSource("classifications")
    void classifiesExactTypedBitsOnce(DataType type, long bits,
            CpuKernelIr.PowerRealization expected) {
        assertEquals(expected, new CpuScalarPowerAnalysis().analyze(
                new CpuKernelIr.ScalarImmediate(type, bits)));
    }

    @Test void rejectsMissingAndNonFloatingExponents() {
        var analysis = new CpuScalarPowerAnalysis();
        assertAll(
                () -> assertEquals("exponent", assertThrows(NullPointerException.class,
                        () -> analysis.analyze(null)).getMessage()),
                () -> assertThrows(IllegalArgumentException.class, () -> analysis.analyze(
                        new CpuKernelIr.ScalarImmediate(DataType.INT32, 1))));
    }

    private static Stream<Arguments> classifications() {
        return Stream.of(
                f32(+0.0f, CpuKernelIr.PowerRealization.POSITIVE_ONE),
                f32(-0.0f, CpuKernelIr.PowerRealization.POSITIVE_ONE),
                f32(1.0f, CpuKernelIr.PowerRealization.IDENTITY),
                f32(2.0f, CpuKernelIr.PowerRealization.SQUARE),
                f32(-1.0f, CpuKernelIr.PowerRealization.RECIPROCAL),
                f32(0.5f, CpuKernelIr.PowerRealization.DIRECT),
                f32(3.0f, CpuKernelIr.PowerRealization.DIRECT),
                f32(-2.0f, CpuKernelIr.PowerRealization.DIRECT),
                f32(Float.POSITIVE_INFINITY, CpuKernelIr.PowerRealization.DIRECT),
                Arguments.of(DataType.FLOAT32, 0x7fc0_0001L,
                        CpuKernelIr.PowerRealization.DIRECT),
                f64(+0.0d, CpuKernelIr.PowerRealization.POSITIVE_ONE),
                f64(-0.0d, CpuKernelIr.PowerRealization.POSITIVE_ONE),
                f64(1.0d, CpuKernelIr.PowerRealization.IDENTITY),
                f64(2.0d, CpuKernelIr.PowerRealization.SQUARE),
                f64(-1.0d, CpuKernelIr.PowerRealization.RECIPROCAL),
                f64(0.5d, CpuKernelIr.PowerRealization.DIRECT),
                f64(3.0d, CpuKernelIr.PowerRealization.DIRECT),
                f64(-2.0d, CpuKernelIr.PowerRealization.DIRECT),
                f64(Double.NEGATIVE_INFINITY, CpuKernelIr.PowerRealization.DIRECT),
                Arguments.of(DataType.FLOAT64, 0x7ff8_0000_0000_0001L,
                        CpuKernelIr.PowerRealization.DIRECT));
    }

    private static Arguments f32(float value, CpuKernelIr.PowerRealization realization) {
        return Arguments.of(DataType.FLOAT32,
                Float.floatToRawIntBits(value) & 0xffff_ffffL, realization);
    }

    private static Arguments f64(double value, CpuKernelIr.PowerRealization realization) {
        return Arguments.of(DataType.FLOAT64, Double.doubleToRawLongBits(value), realization);
    }
}
