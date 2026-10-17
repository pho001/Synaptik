package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuBatchNormInferenceIrTest {
    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }

    @Test void locksFivePositionsPromotionEpsilonAxisMapAndRangeIdentity() {
        var ir = new CpuBatchNormInferenceIr(List.of(DataType.BFLOAT16, DataType.FLOAT32,
                DataType.BFLOAT16, DataType.FLOAT32, DataType.BFLOAT16), DataType.FLOAT32,
                Float.floatToRawIntBits(1e-5f) & 0xffff_ffffL, 3, 1, 1,
                CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE, List.of(0, 1, 2, 1, 2),
                List.of(access(CpuAccessPlan.AccessKind.READ, 3),
                        access(CpuAccessPlan.AccessKind.READ, 1),
                        access(CpuAccessPlan.AccessKind.READ, 1)),
                access(CpuAccessPlan.AccessKind.WRITE, 3));
        var other = ir.withRangeForm(CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE);
        assertAll(() -> assertNotEquals(ir.structuralKey(), other.structuralKey()),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity().contains("axis=1")),
                () -> assertTrue(other.encodedKernelIr().familyIdentity()
                        .endsWith("range=NON_CHANNEL_RANGE")),
                () -> assertEquals(4, ir.encodedKernelIr().values().size()));
    }

    @Test void rejectsInvalidPromotionAndRepeatedBoundaryTypes() {
        assertThrows(IllegalArgumentException.class, () -> new CpuBatchNormInferenceIr(
                java.util.Collections.nCopies(5, DataType.FLOAT32), DataType.FLOAT64,
                Double.doubleToRawLongBits(1e-5), 2, 1, 1,
                CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE, List.of(0, 1, 2, 3, 4),
                java.util.Collections.nCopies(5, access(CpuAccessPlan.AccessKind.READ, 1)),
                access(CpuAccessPlan.AccessKind.WRITE, 2)));
    }
}
