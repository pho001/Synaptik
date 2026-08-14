package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuAggregateIrTest {
    @Test void canonicalIdentityIncludesKindFormMembershipRetentionAndPolicies() {
        var read = plan(CpuAccessPlan.AccessKind.READ, 3);
        var write = plan(CpuAccessPlan.AccessKind.WRITE, 1);
        var first = new CpuAggregateIr(CpuAggregateIr.Kind.MIN, DataType.FLOAT32,
                CpuAggregateIr.Form.MULTI_AXIS, new int[] {0, 2}, false, read, write,
                CpuAggregateIr.FIRST_LOGICAL_NAN_AND_SIGNED_ZERO,
                CpuAggregateIr.COMPLETE_OUTPUT_CELLS, CpuAggregateIr.ZERO_WORKSPACE);
        var same = new CpuAggregateIr(CpuAggregateIr.Kind.MIN, DataType.FLOAT32,
                CpuAggregateIr.Form.MULTI_AXIS, new int[] {0, 2}, false, read, write, 1, 1, 0);
        var retained = new CpuAggregateIr(CpuAggregateIr.Kind.MIN, DataType.FLOAT32,
                CpuAggregateIr.Form.MULTI_AXIS, new int[] {0, 2}, true, read,
                plan(CpuAccessPlan.AccessKind.WRITE, 3), 1, 1, 0);
        assertAll(() -> assertEquals(first.structuralKey(), same.structuralKey()),
                () -> assertNotEquals(first.structuralKey(), retained.structuralKey()),
                () -> assertArrayEquals(new int[] {0, 2}, first.selectedAxes()));
    }

    @Test void rejectsUnsortedAxesAndTypeKindMismatch() {
        assertThrows(IllegalArgumentException.class, () -> new CpuAggregateIr(
                CpuAggregateIr.Kind.MAX, DataType.FLOAT64, CpuAggregateIr.Form.MULTI_AXIS,
                new int[] {2, 0}, false, plan(CpuAccessPlan.AccessKind.READ, 3),
                plan(CpuAccessPlan.AccessKind.WRITE, 1), 1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new CpuAggregateIr(
                CpuAggregateIr.Kind.ALL, DataType.INT32, CpuAggregateIr.Form.FULL,
                new int[] {0}, false, plan(CpuAccessPlan.AccessKind.READ, 1),
                plan(CpuAccessPlan.AccessKind.WRITE, 0), 1, 1, 0));
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.GENERAL_ODOMETER, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.STRIDED), 0);
    }
}
