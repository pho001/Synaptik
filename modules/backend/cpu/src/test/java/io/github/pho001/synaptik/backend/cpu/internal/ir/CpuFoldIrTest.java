package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuFoldIrTest {
    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }

    @Test void structuralIdentitySeparatesFamilyTypeRankAndAdditionPolicy() {
        var axis = new CpuFoldIr(CpuFoldIr.Family.FOLD_AXIS, DataType.FLOAT32,
                access(CpuAccessPlan.AccessKind.READ, 2),
                access(CpuAccessPlan.AccessKind.WRITE, 1),
                CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION);
        var image = new CpuFoldIr(CpuFoldIr.Family.FOLD2D, DataType.FLOAT32,
                access(CpuAccessPlan.AccessKind.READ, 3),
                access(CpuAccessPlan.AccessKind.WRITE, 4),
                CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION);
        var integral = new CpuFoldIr(CpuFoldIr.Family.FOLD_AXIS, DataType.INT64,
                access(CpuAccessPlan.AccessKind.READ, 2),
                access(CpuAccessPlan.AccessKind.WRITE, 1),
                CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION);
        assertAll(() -> assertNotEquals(axis.structuralKey(), image.structuralKey()),
                () -> assertNotEquals(axis.structuralKey(), integral.structuralKey()),
                () -> assertTrue(axis.encodedKernelIr().familyIdentity().contains("addition=1")),
                () -> assertEquals(List.of(DataType.FLOAT32, DataType.FLOAT32),
                        axis.encodedKernelIr().values().stream()
                                .map(CpuKernelIr.Value::dataType).toList()));
    }

    @Test void rejectsBoolIntegral2dWrongAccessAndUnknownPolicy() {
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuFoldIr(
                        CpuFoldIr.Family.FOLD_AXIS, DataType.BOOL,
                        access(CpuAccessPlan.AccessKind.READ, 2),
                        access(CpuAccessPlan.AccessKind.WRITE, 1), 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuFoldIr(
                        CpuFoldIr.Family.FOLD2D, DataType.INT32,
                        access(CpuAccessPlan.AccessKind.READ, 3),
                        access(CpuAccessPlan.AccessKind.WRITE, 4), 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuFoldIr(
                        CpuFoldIr.Family.FOLD_AXIS, DataType.FLOAT64,
                        access(CpuAccessPlan.AccessKind.WRITE, 2),
                        access(CpuAccessPlan.AccessKind.WRITE, 1), 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuFoldIr(
                        CpuFoldIr.Family.FOLD_AXIS, DataType.FLOAT64,
                        access(CpuAccessPlan.AccessKind.READ, 2),
                        access(CpuAccessPlan.AccessKind.WRITE, 1), 2)));
    }
}
