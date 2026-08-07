package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class CpuReferenceDifferentialTest {
    @Test void preservesClassificationsSignedZeroAndOracleTolerance() {
        assertAll(
                () -> assertTrue(Double.isNaN(CpuScalarReferenceKernel.erf(Double.NaN))),
                () -> assertEquals(1.0, CpuScalarReferenceKernel.erf(Double.POSITIVE_INFINITY)),
                () -> assertEquals(-1.0, CpuScalarReferenceKernel.erf(Double.NEGATIVE_INFINITY)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0),
                        Double.doubleToRawLongBits(CpuScalarReferenceKernel.erf(-0.0))),
                () -> assertTrue(Double.isNaN(CpuScalarReferenceKernel.gelu(Double.NaN))),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        CpuScalarReferenceKernel.gelu(Double.POSITIVE_INFINITY)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.gelu(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0),
                        Double.doubleToRawLongBits(CpuScalarReferenceKernel.gelu(-0.0))),
                () -> assertEquals(0.8427007929497149, CpuScalarReferenceKernel.erf(1.0), 2e-7));
        assertAll(
                () -> assertEquals(0.0d, CpuScalarReferenceKernel.sigmoid(Double.NEGATIVE_INFINITY)),
                () -> assertEquals(1.0d, CpuScalarReferenceKernel.sigmoid(Double.POSITIVE_INFINITY)),
                () -> assertEquals(0.5d, CpuScalarReferenceKernel.sigmoid(-0.0d)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.geluTanhApproximation(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.silu(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.silu(-0.0d))));
        double[][] oracle = {
                {-3.0, -0.00404969409489031}, {-1.0, -0.15865525393145707},
                {-0.25, -0.100323432704662}, {0.0, 0.0},
                {0.25, 0.149676567295338}, {1.0, 0.8413447460685429},
                {3.0, 2.99595030590511}
        };
        for (double[] pair : oracle) assertEquals(pair[1],
                CpuScalarReferenceKernel.gelu(pair[0]),
                2e-7 * Math.max(1.0, Math.abs(pair[1])));
    }
}
