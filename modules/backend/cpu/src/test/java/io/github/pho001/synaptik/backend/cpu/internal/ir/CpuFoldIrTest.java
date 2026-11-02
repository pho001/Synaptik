package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuFoldIrTest {
    @Test void lowAccumulatorVersionChangesOnlyLowFoldCanonicalIdentity() {
        for (var family : CpuFoldIr.Family.values()) for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            var ir = new CpuFoldIr(family, type,
                    access(CpuAccessPlan.AccessKind.READ, family == CpuFoldIr.Family.FOLD_AXIS ? 2 : 3),
                    access(CpuAccessPlan.AccessKind.WRITE, family == CpuFoldIr.Family.FOLD_AXIS ? 1 : 4), 1);
            boolean low = type == DataType.BFLOAT16 || type == DataType.FLOAT16;
            assertEquals("fold:" + family + ":addition=1" + (low ? ":low-fp32-accumulation-v1" : ""), ir.encodedKernelIr().familyIdentity());
        }
    }
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
