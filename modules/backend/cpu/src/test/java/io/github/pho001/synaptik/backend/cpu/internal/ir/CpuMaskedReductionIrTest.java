package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuMaskedReductionIrTest {
    @Test void identityIncludesMeaningAxisBroadcastTopologyAccessAndExactState() {
        var data = access(CpuAccessPlan.AccessKind.READ, 2);
        var mask = access(CpuAccessPlan.AccessKind.READ, 1);
        var output = access(CpuAccessPlan.AccessKind.WRITE, 1);
        var base = new CpuMaskedReductionIr(CpuMaskedReductionIr.Kind.SUM, DataType.FLOAT32,
                1, 1, new boolean[] {false, false}, 4, 4, 40, data, mask, output);
        assertAll(
                () -> assertEquals(DataType.BOOL,
                        base.encodedKernelIr().values().get(1).dataType()),
                () -> assertNotEquals(base.structuralKey(), new CpuMaskedReductionIr(
                        CpuMaskedReductionIr.Kind.MEAN, DataType.FLOAT32, 1, 1,
                        new boolean[] {false, false}, 4, 4, 40, data, mask, output)
                        .structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuMaskedReductionIr(
                        CpuMaskedReductionIr.Kind.SUM, DataType.FLOAT32, 1, 1,
                        new boolean[] {false, true}, 4, 4, 40, data, mask, output)
                        .structuralKey()),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuMaskedReductionIr(CpuMaskedReductionIr.Kind.SUM, DataType.INT32,
                                1, 1, new boolean[] {false, false}, 4, 4, 40,
                                data, mask, output)));
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }
}
