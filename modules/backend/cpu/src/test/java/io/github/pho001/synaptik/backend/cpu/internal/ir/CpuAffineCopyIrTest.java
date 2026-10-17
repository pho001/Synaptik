package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuAffineCopyIrTest {
    @Test void exposesASealedFamilyRoleAndDeterministicEncodedIdentity() {
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuAffineCopyIr(DataType.BFLOAT16, read, write,
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.SELECT,
                        2, 1, List.of(0))),
                CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        assertAll(
                () -> assertTrue(CpuPortableKernelIr.class.isSealed()),
                () -> assertEquals(ir.structuralKey(), ir.encodedKernelIr().structuralKey()),
                () -> assertTrue(ir.encodedKernelIr().instructions().isEmpty()),
                () -> assertEquals(2, ir.encodedKernelIr().values().size()));
        var otherAxis = new CpuAffineCopyIr(DataType.BFLOAT16, read, write,
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.SELECT,
                        2, 1, List.of(1))), CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        assertNotEquals(ir.structuralKey(), otherAxis.structuralKey());
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }
}
