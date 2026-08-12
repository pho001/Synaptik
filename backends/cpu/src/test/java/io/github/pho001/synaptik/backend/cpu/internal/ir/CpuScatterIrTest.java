package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuScatterIrTest {
    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    @Test void structuralIdentitySeparatesFamilyReductionMappingTypeAndScratch() {
        var base = new CpuScatterIr(CpuScatterIr.Family.SCATTER_ELEMENTS,
                ScatterReduction.ADD, List.of(0,1,2),
                List.of(DataType.FLOAT32,DataType.INT32,DataType.FLOAT32,DataType.FLOAT32),
                List.of(access(CpuAccessPlan.AccessKind.READ),access(CpuAccessPlan.AccessKind.READ),
                        access(CpuAccessPlan.AccessKind.READ),access(CpuAccessPlan.AccessKind.WRITE)),0);
        var nd = new CpuScatterIr(CpuScatterIr.Family.SCATTER_ND,
                ScatterReduction.ADD, base.occurrenceToBoundary(),base.boundaryTypes(),
                base.boundaryAccesses(),0);
        var mul = new CpuScatterIr(CpuScatterIr.Family.SCATTER_ELEMENTS,
                ScatterReduction.MUL,base.occurrenceToBoundary(),base.boundaryTypes(),
                base.boundaryAccesses(),1);
        assertAll(() -> assertNotEquals(base.structuralKey(),nd.structuralKey()),
                () -> assertNotEquals(base.structuralKey(),mul.structuralKey()),
                () -> assertTrue(mul.encodedKernelIr().familyIdentity().contains("scratch=1")),
                () -> assertEquals(4,mul.encodedKernelIr().values().size()));
    }

    @Test void rejectsWrongArityFixedAddReductionAndIneligibleScratch() {
        var accesses=List.of(access(CpuAccessPlan.AccessKind.READ),access(CpuAccessPlan.AccessKind.WRITE));
        assertAll(() -> assertThrows(IllegalArgumentException.class,()->new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ADD,ScatterReduction.MUL,List.of(0,0,0),
                        List.of(DataType.INT32,DataType.INT32),accesses,0)),
                () -> assertThrows(IllegalArgumentException.class,()->new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS,ScatterReduction.NONE,List.of(0,0),
                        List.of(DataType.INT32,DataType.INT32),accesses,0)),
                () -> assertThrows(IllegalArgumentException.class,()->new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS,ScatterReduction.MUL,List.of(0,0,0),
                        List.of(DataType.INT32,DataType.INT32),accesses,1)));
    }

    @Test void rejectsMalformedOccurrenceAccessAndTypeAssignments() {
        var readsAndWrite = List.of(access(CpuAccessPlan.AccessKind.READ),
                access(CpuAccessPlan.AccessKind.READ), access(CpuAccessPlan.AccessKind.READ),
                access(CpuAccessPlan.AccessKind.WRITE));
        var types = List.of(DataType.FLOAT32, DataType.INT32, DataType.FLOAT32,
                DataType.FLOAT32);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS, ScatterReduction.ADD,
                        List.of(0, 1, 3), types, readsAndWrite, 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS, ScatterReduction.ADD,
                        List.of(0, 1, 1), types, readsAndWrite, 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS, ScatterReduction.ADD,
                        List.of(0, 1, 2), types, List.of(access(CpuAccessPlan.AccessKind.WRITE),
                                access(CpuAccessPlan.AccessKind.READ),
                                access(CpuAccessPlan.AccessKind.READ),
                                access(CpuAccessPlan.AccessKind.WRITE)), 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS, ScatterReduction.ADD,
                        List.of(0, 1, 2), List.of(DataType.FLOAT32, DataType.BOOL,
                                DataType.FLOAT32, DataType.FLOAT32), readsAndWrite, 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuScatterIr(
                        CpuScatterIr.Family.SCATTER_ELEMENTS, ScatterReduction.NONE,
                        List.of(0, 1, 2), List.of(DataType.INT32, DataType.INT32,
                                DataType.INT64, DataType.INT32), readsAndWrite, 0)));
    }
}
