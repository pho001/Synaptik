package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import org.junit.jupiter.api.Test;

class CpuSoftmaxReferenceTest {
    @Test void preservesLogicalNormalizationAxes() {
        double[] result = CpuSoftmaxReferenceKernel.evaluate(SoftmaxKind.SOFTMAX,
                DataType.FLOAT64, new double[] {1, 2, 3, 4}, new long[] {2, 2}, 0,
                new long[] {2, 1}, 1);
        assertAll(() -> assertEquals(1.0, result[0] + result[1], 2e-15),
                () -> assertEquals(1.0, result[2] + result[3], 2e-15));
    }

    @Test void bothWorkingTypesAdmitTiedFiniteExtremesOnStridedNonLastAxes() {
        for (DataType type : new DataType[] {DataType.FLOAT32, DataType.FLOAT64})
            for (SoftmaxKind kind : SoftmaxKind.values()) {
                double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                double[] input = new double[16];
                java.util.Arrays.fill(input, Double.NaN);
                for (int index : new int[] {1, 3, 5, 9, 11, 13})
                    input[index] = index == 1 || index == 9 ? -maximum : maximum;
                double[] result = CpuSoftmaxReferenceKernel.evaluate(kind, type, input,
                        new long[] {3, 2}, 1, new long[] {2, 8}, 0);
                double half = kind == SoftmaxKind.SOFTMAX ? 0.5
                        : type == DataType.FLOAT32 ? (float) -StrictMath.log(2) : -StrictMath.log(2);
                assertArrayEquals(new double[] {kind == SoftmaxKind.SOFTMAX ? 0 : Double.NEGATIVE_INFINITY,
                        kind == SoftmaxKind.SOFTMAX ? 0 : Double.NEGATIVE_INFINITY,
                        half, half, half, half}, result);
                assertArrayEquals(new double[] {kind == SoftmaxKind.SOFTMAX ? 1 : 0},
                        CpuSoftmaxReferenceKernel.evaluate(kind, type, new double[] {-maximum},
                                new long[] {1}, 0, new long[] {1}, 0));
                assertEquals(0, CpuSoftmaxReferenceKernel.evaluate(kind, type, new double[0],
                        new long[] {3, 0}, 0, new long[] {1, 3}, 0).length);
            }
    }

    @Test void bfloatWorkingShiftOverflowKeepsModelResultClasses() {
        double maximum = Float.intBitsToFloat(0x7f7f0000);
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            double[] actual = CpuSoftmaxReferenceKernel.evaluate(
                    kind, DataType.BFLOAT16, new double[] {-maximum, maximum},
                    new long[] {2}, 0, new long[] {1}, 0);
            assertArrayEquals(kind == SoftmaxKind.SOFTMAX
                    ? new double[]{0.0d, 1.0d} : new double[]{Double.NEGATIVE_INFINITY, 0.0d},
                    actual);
        }
    }
}
