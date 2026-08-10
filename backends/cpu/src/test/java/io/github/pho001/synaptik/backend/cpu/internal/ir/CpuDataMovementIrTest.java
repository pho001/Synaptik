package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuDataMovementIrTest {
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

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }
}
