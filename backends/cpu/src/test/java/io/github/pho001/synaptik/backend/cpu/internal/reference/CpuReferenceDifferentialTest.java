package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;

class CpuReferenceDifferentialTest {
    @Test void affineReferencePreservesOpaqueBfloat16AddressPairs() {
        var lowered = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering()
                .lower(CpuAffineLayoutLoweringTest.select(DataType.BFLOAT16, List.of()));
        var encoded = lowered.kernelIr();
        var affine = new CpuAffineCopyIr(DataType.BFLOAT16,
                encoded.values().get(0).accessPlan(), encoded.values().get(1).accessPlan(),
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.SELECT,
                        2, 1, List.of(0))),
                CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        short[] source = {1, (short)0x7fc1, 2, 3, (short)0xff80, 4, 5, (short)0x8000, 6};
        short[] output = new short[8];
        CpuScalarReferenceKernel.execute(affine, lowered.affineAddressPairs(), List.of(
                new CpuBufferArgument.Shorts(source, 0, 18, true),
                new CpuBufferArgument.Shorts(output, 0, 16, false)), 0, 3);
        assertAll(() -> assertEquals(source[1], output[1]),
                () -> assertEquals(source[4], output[4]),
                () -> assertEquals(source[7], output[7]));
    }
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
