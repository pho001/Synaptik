package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuScanIrTest {
    @Test void structuralIdentityIncludesEveryCodeShapingScanFact() {
        var read = access(CpuAccessPlan.AccessKind.READ);
        var write = access(CpuAccessPlan.AccessKind.WRITE);
        var base = new CpuScanIr(CpuScanIr.Kind.CUM_SUM, DataType.FLOAT32, 1, false, false,
                read, write, CpuScanIr.SEQUENTIAL_TYPED_ROUNDING);
        assertAll(() -> assertNotEquals(base.structuralKey(), new CpuScanIr(CpuScanIr.Kind.CUM_PROD,
                        DataType.FLOAT32, 1, false, false, read, write, 1).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuScanIr(CpuScanIr.Kind.CUM_SUM,
                        DataType.FLOAT32, 0, false, false, read, write, 1).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuScanIr(CpuScanIr.Kind.CUM_SUM,
                        DataType.FLOAT32, 1, true, false, read, write, 1).structuralKey()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScanIr(
                        CpuScanIr.Kind.CUM_SUM, DataType.BOOL, 0, false, false, read, write, 1)));
    }
    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
    }
}
