package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuOrderingIrTest {
    @Test void identitySeparatesFamilyDirectionOrderTypeAndOutputCount() {
        var read = plan(CpuAccessPlan.AccessKind.READ, 2);
        var write = plan(CpuAccessPlan.AccessKind.WRITE, 2);
        var sort = new CpuOrderingIr(CpuOrderingIr.Family.SORT, DataType.FLOAT32, false, true,
                List.of(read, write), CpuOrderingIr.TWO_INDEX_MERGE_REGIONS);
        var argsort = new CpuOrderingIr(CpuOrderingIr.Family.ARGSORT, DataType.FLOAT32, false, true,
                List.of(read, write), CpuOrderingIr.TWO_INDEX_MERGE_REGIONS);
        var top = new CpuOrderingIr(CpuOrderingIr.Family.TOP_K, DataType.FLOAT32, true, false,
                List.of(read, write, write), CpuOrderingIr.TWO_INDEX_MERGE_REGIONS);
        assertAll(() -> assertNotEquals(sort.structuralKey(), argsort.structuralKey()),
                () -> assertNotEquals(sort.structuralKey(), top.structuralKey()),
                () -> assertEquals(3, top.encodedKernelIr().values().size()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuOrderingIr(
                        CpuOrderingIr.Family.TOP_K, DataType.FLOAT32, true, false,
                        List.of(read, write), CpuOrderingIr.TWO_INDEX_MERGE_REGIONS)));
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }
}
