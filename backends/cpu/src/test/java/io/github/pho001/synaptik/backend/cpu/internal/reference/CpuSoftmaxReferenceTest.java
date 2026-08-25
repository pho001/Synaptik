package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import org.junit.jupiter.api.Test;

class CpuSoftmaxReferenceTest {
    @Test void preservesAxesAndRejectsTheUnsettledDomain() {
        double[] result = CpuSoftmaxReferenceKernel.evaluate(SoftmaxKind.SOFTMAX,
                DataType.FLOAT64, new double[] {1, 2, 3, 4}, new long[] {2, 2}, 0,
                new long[] {2, 1}, 1);
        assertAll(() -> assertEquals(1.0, result[0] + result[1], 2e-15),
                () -> assertEquals(1.0, result[2] + result[3], 2e-15),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        CpuSoftmaxReferenceKernel.evaluate(SoftmaxKind.LOG_SOFTMAX,
                                DataType.FLOAT64, new double[] {Double.POSITIVE_INFINITY},
                                new long[] {1}, 0, new long[] {1}, 0)));
    }
}
