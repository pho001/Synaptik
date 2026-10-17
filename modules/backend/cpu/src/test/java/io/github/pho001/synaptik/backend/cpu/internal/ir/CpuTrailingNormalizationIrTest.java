package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuTrailingNormalizationIrTest {
    @Test void identityRetainsExactFormTypesEpsilonPassesAndBoundaryMap() {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        var parameter = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        var ir = new CpuTrailingNormalizationIr(CpuTrailingNormalizationIr.Kind.LAYER,
                CpuTrailingNormalizationIr.Form.LAYER_AFFINE,
                List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.BFLOAT16), DataType.FLOAT32,
                Float.floatToRawIntBits(1e-5f) & 0xffff_ffffL, 1, 1, 3, 3, 6, 56,
                List.of(0, 1, 1), List.of(read, parameter), write);
        String identity = ir.encodedKernelIr().familyIdentity();
        assertAll(() -> assertTrue(identity.contains(":form=LAYER_AFFINE:")),
                () -> assertTrue(identity.contains(":passes=3:")),
                () -> assertTrue(identity.contains(":map=[0, 1, 1]:")),
                () -> assertEquals(ir.structuralKey(), ir.encodedKernelIr().structuralKey()));
    }

    @Test void rejectsNonCanonicalBoundaryMapsPromotionAndEpsilonIdentity() {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuTrailingNormalizationIr(CpuTrailingNormalizationIr.Kind.RMS,
                                CpuTrailingNormalizationIr.Form.RMS_SCALED,
                                List.of(DataType.FLOAT32, DataType.FLOAT64), DataType.FLOAT32,
                                Float.floatToRawIntBits(1e-5f) & 0xffff_ffffL, 1, 1, 2, 3,
                                0, 0, List.of(0, 1), List.of(read, read), write)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuTrailingNormalizationIr(CpuTrailingNormalizationIr.Kind.RMS,
                                CpuTrailingNormalizationIr.Form.RMS_SCALED,
                                List.of(DataType.FLOAT32, DataType.FLOAT32), DataType.FLOAT32,
                                Float.floatToRawIntBits(1e-5f) & 0xffff_ffffL, 1, 1, 2, 3,
                                0, 0, List.of(1, 0), List.of(read, read), write)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuTrailingNormalizationIr(CpuTrailingNormalizationIr.Kind.RMS,
                                CpuTrailingNormalizationIr.Form.RMS,
                                List.of(DataType.FLOAT32), DataType.FLOAT32,
                                Float.floatToRawIntBits(-1.0f) & 0xffff_ffffL, 1, 1, 2, 3,
                                0, 0, List.of(0), List.of(read), write)));
    }
}
