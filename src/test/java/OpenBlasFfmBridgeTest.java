import backend.blas.OpenBlasFfmBridge;
import backend.cpu.kernels.CpuDTypeOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OpenBlasFfmBridgeTest {
    @Test
    void bundledOrConfiguredOpenBlasProvidesRequiredGemmSymbols() {
        assertTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        double[] a64 = {1.0d, 2.0d, 3.0d, 4.0d};
        double[] b64 = {5.0d, 6.0d, 7.0d, 8.0d};
        double[] c64 = new double[4];
        OpenBlasFfmBridge.dgemmRowMajorNoTrans(2, 2, 2, 1.0d, a64, 2, b64, 2, 0.0d, c64, 2);
        assertArrayEquals(new double[]{19.0d, 22.0d, 43.0d, 50.0d}, c64, 1e-12);

        float[] a32 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b32 = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] c32 = new float[4];
        OpenBlasFfmBridge.sgemmRowMajorNoTrans(2, 2, 2, 1.0f, a32, 2, b32, 2, 0.0f, c32, 2);
        assertArrayEquals(new float[]{19.0f, 22.0f, 43.0f, 50.0f}, c32, 1e-6f);
    }

    @Test
    void bundledOrConfiguredOpenBlasProvidesBFloat16GemmWhenAdvertised() {
        assertTrue(OpenBlasFfmBridge.isBFloat16GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        short[] a = {
                CpuDTypeOps.toBFloat16Bits(1.0f),
                CpuDTypeOps.toBFloat16Bits(2.0f),
                CpuDTypeOps.toBFloat16Bits(3.0f),
                CpuDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] b = {
                CpuDTypeOps.toBFloat16Bits(5.0f),
                CpuDTypeOps.toBFloat16Bits(6.0f),
                CpuDTypeOps.toBFloat16Bits(7.0f),
                CpuDTypeOps.toBFloat16Bits(8.0f)
        };
        float[] c = new float[4];

        OpenBlasFfmBridge.sbgemmRowMajorNoTrans(2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2);

        assertArrayEquals(new float[]{19.0f, 22.0f, 43.0f, 50.0f}, c, 1e-6f);
    }
}
