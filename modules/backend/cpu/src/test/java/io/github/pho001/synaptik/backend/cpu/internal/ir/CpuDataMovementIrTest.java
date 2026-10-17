package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuDataMovementIrTest {
    @Test void sliceUpdateIdentityIncludesRankAndOccurrenceMapOnly() {
        var read = access(CpuAccessPlan.AccessKind.READ, 1);
        var write = access(CpuAccessPlan.AccessKind.WRITE, 1);
        var distinct = new CpuDataMovementIr(DataType.INT64,
                new CpuDataMovementIr.SliceUpdatePlan(1, List.of(0, 1)),
                List.of(read, read), write);
        var deduplicated = new CpuDataMovementIr(DataType.INT64,
                new CpuDataMovementIr.SliceUpdatePlan(1, List.of(0, 0)),
                List.of(read), write);
        assertAll(
                () -> assertEquals("SLICE_UPDATE", distinct.plan().family()),
                () -> assertNotEquals(distinct.structuralKey(), deduplicated.structuralKey()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuDataMovementIr.SliceUpdatePlan(1, List.of(0))));
    }

    @Test void retainsClosedVariantOccurrenceOrderAndExactPadBits() {
        var read = access(CpuAccessPlan.AccessKind.READ, 2);
        var write = access(CpuAccessPlan.AccessKind.WRITE, 2);
        var concat = new CpuDataMovementIr(DataType.INT32,
                new CpuDataMovementIr.ConcatPlan(2, List.of(0, 1, 0)),
                List.of(read, read), write);
        var pad = new CpuDataMovementIr(DataType.FLOAT32,
                new CpuDataMovementIr.PadPlan(2, 0x7fc0_0042L), List.of(read), write);
        assertAll(
                () -> assertEquals(List.of(0, 1, 0), concat.plan().occurrenceToBoundary()),
                () -> assertTrue(concat.structuralKey().matches("[0-9a-f]{64}")),
                () -> assertNotEquals(concat.structuralKey(), new CpuDataMovementIr(DataType.INT32,
                        new CpuDataMovementIr.ConcatPlan(2, List.of(0, 0, 1)),
                        List.of(read, read), write).structuralKey()),
                () -> assertNotEquals(pad.structuralKey(), new CpuDataMovementIr(DataType.FLOAT32,
                        new CpuDataMovementIr.PadPlan(2, 0x7fc0_0043L), List.of(read), write)
                        .structuralKey()));
    }

    @Test void rejectsInvalidBoundaryAndOccurrenceStructures() {
        var read = access(CpuAccessPlan.AccessKind.READ, 1);
        var write = access(CpuAccessPlan.AccessKind.WRITE, 1);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuDataMovementIr(
                        DataType.INT64, new CpuDataMovementIr.ConcatPlan(1, List.of(1)),
                        List.of(read), write)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuDataMovementIr.StackPlan(1, List.of())),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuDataMovementIr(
                        DataType.INT64, new CpuDataMovementIr.TilePlan(1), List.of(write), write)));
    }

    @Test void windowIdentityIncludesFamilyRanksAccessAndPaddingBits() {
        var input4 = access(CpuAccessPlan.AccessKind.READ, 4);
        var output3 = access(CpuAccessPlan.AccessKind.WRITE, 3);
        var direct = new CpuDataMovementIr(DataType.FLOAT32,
                new CpuDataMovementIr.Unfold2dPlan(3, 0), List.of(input4), output3);
        var negativeZero = new CpuDataMovementIr(DataType.FLOAT32,
                new CpuDataMovementIr.Unfold2dPlan(3, 0x8000_0000L), List.of(input4), output3);
        var axis = new CpuDataMovementIr(DataType.FLOAT32,
                new CpuDataMovementIr.UnfoldAxisPlan(3),
                List.of(access(CpuAccessPlan.AccessKind.READ, 2)), output3);
        assertAll(
                () -> assertNotEquals(direct.structuralKey(), negativeZero.structuralKey()),
                () -> assertNotEquals(direct.structuralKey(), axis.structuralKey()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuDataMovementIr(
                        DataType.FLOAT32, new CpuDataMovementIr.Unfold2dPlan(3, 0),
                        List.of(access(CpuAccessPlan.AccessKind.READ, 3)), output3)));
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }
}
