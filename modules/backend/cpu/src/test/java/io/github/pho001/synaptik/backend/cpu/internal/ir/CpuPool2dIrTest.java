package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuPool2dIrTest {
    @Test
    void retainsOnlyFamilyTypeRealizationAndAccess() {
        var read =
                new CpuAccessPlan(
                        CpuAccessPlan.AccessKind.READ,
                        CpuAccessPlan.Regime.DENSE_LINEAR,
                        4,
                        List.of(
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS),
                        4);
        var write =
                new CpuAccessPlan(
                        CpuAccessPlan.AccessKind.WRITE,
                        CpuAccessPlan.Regime.DENSE_LINEAR,
                        4,
                        List.of(
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS,
                                CpuAccessPlan.AxisRole.CONTIGUOUS),
                        4);
        var ir =
                new CpuPool2dIr(
                        CpuPool2dIr.Kind.MAX,
                        DataType.BFLOAT16,
                        CpuPool2dIr.Realization.DIRECT_SCALAR,
                        read,
                        write);
        assertTrue(ir.encodedKernelIr().familyIdentity().startsWith("pool2d:kind=MAX:"));
        assertEquals(2, ir.encodedKernelIr().values().size());
    }
}
