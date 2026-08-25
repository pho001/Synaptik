package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuSoftmaxIrTest {
    @Test void retainsFirstClassKindAxisPassAndAccessIdentity() {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        var softmax = new CpuSoftmaxIr(SoftmaxKind.SOFTMAX, DataType.FLOAT32, 1, 1, 3,
                read, write);
        var log = new CpuSoftmaxIr(SoftmaxKind.LOG_SOFTMAX, DataType.FLOAT32, 1, 1, 3,
                read, write);
        assertAll(() -> assertNotEquals(softmax.structuralKey(), log.structuralKey()),
                () -> assertTrue(softmax.encodedKernelIr().familyIdentity()
                        .startsWith("softmax:SOFTMAX:")),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuSoftmaxIr(
                        SoftmaxKind.SOFTMAX, DataType.INT32, 1, 1, 3, read, write)));
    }
}
