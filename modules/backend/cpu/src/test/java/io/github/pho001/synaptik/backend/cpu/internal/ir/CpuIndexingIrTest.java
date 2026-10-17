package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuIndexingIrTest {
    @Test void structuralIdentityIncludesFamilyMapTypesAndAccessRanks() {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var gather = new CpuIndexingIr(CpuIndexingIr.Family.GATHER, List.of(0, 1),
                List.of(DataType.FLOAT32, DataType.INT64, DataType.FLOAT32),
                List.of(read, read, write));
        var elements = new CpuIndexingIr(CpuIndexingIr.Family.GATHER_ELEMENTS, List.of(0, 1),
                List.of(DataType.FLOAT32, DataType.INT64, DataType.FLOAT32),
                List.of(read, read, write));
        assertAll(() -> assertNotEquals(gather.structuralKey(), elements.structuralKey()),
                () -> assertTrue(gather.encodedKernelIr().familyIdentity().startsWith("indexing:GATHER")),
                () -> assertEquals(3, gather.encodedKernelIr().values().size()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuIndexingIr(
                        CpuIndexingIr.Family.ONE_HOT, List.of(0, 1),
                        List.of(DataType.INT32, DataType.BOOL), List.of(read, write))));
    }
}
