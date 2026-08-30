package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Collections;
import org.junit.jupiter.api.Test;

class CpuPool3dIrTest {
    @Test void retainsOnlyRankSpecificCodeShapingFacts() {
        var roles=Collections.nCopies(5,CpuAccessPlan.AxisRole.CONTIGUOUS);
        var ir=new CpuPool3dIr(CpuPool3dIr.Kind.MAX,DataType.BFLOAT16,
                CpuPool3dIr.Realization.DIRECT_SCALAR,
                new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.DENSE_LINEAR,5,roles,5),
                new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,CpuAccessPlan.Regime.DENSE_LINEAR,5,roles,5));
        assertTrue(ir.encodedKernelIr().familyIdentity().startsWith("pool3d:kind=MAX:"));
        assertEquals(2,ir.encodedKernelIr().values().size());
    }
}
