package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuRandomIrTest {
    @Test void locksAlgorithmNumericAndBoundaryIdentity() {
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuRandomIr(CpuRandomIr.Family.DROPOUT, DataType.FLOAT64, 0, 0,
                Double.doubleToRawLongBits(.5), List.of(read, read, write, write, write));
        assertAll(() -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.GENERATOR_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains("keyBias=9e3779b97f4a7c15")),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains("mixMultiplier1=bf58476d1ce4e5b9")),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains("mixMultiplier2=94d049bb133111eb")),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.COUNTER_MAPPING_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity().contains(CpuRandomIr.UNIFORM_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.THRESHOLD_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.NUMERIC_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.MASK_POLICY_ID)),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity()
                        .contains(CpuRandomIr.STATE_POLICY_ID)),
                () -> assertEquals(List.of(DataType.FLOAT64, DataType.INT64, DataType.FLOAT64,
                        DataType.BOOL, DataType.INT64), ir.encodedKernelIr().values().stream()
                        .map(CpuKernelIr.Value::dataType).toList()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuRandomIr(
                        CpuRandomIr.Family.DROPOUT, DataType.BFLOAT16, 0, 0, 0,
                        List.of(read, read, write, write, write))));
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }
}
