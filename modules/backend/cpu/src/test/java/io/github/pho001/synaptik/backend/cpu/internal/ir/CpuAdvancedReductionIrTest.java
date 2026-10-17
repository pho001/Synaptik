package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuAdvancedReductionIrTest {
    @Test void snapshotsOrderedAxesAndEncodesEveryStructuralNumericalFact() {
        int[] axes = {2, 0}; boolean[] selected = {true, false, true};
        var input = access(CpuAccessPlan.AccessKind.READ, 3);
        var output = access(CpuAccessPlan.AccessKind.WRITE, 1);
        var ir = new CpuAdvancedReductionIr(CpuAdvancedReductionIr.Kind.VARIANCE,
                DataType.FLOAT32, axes, selected, false, 1, 1, 2, 8, 7, 64, input, output);
        axes[0] = 1; selected[0] = false;
        assertAll(() -> assertArrayEquals(new int[] {2, 0}, ir.orderedAxes()),
                () -> assertArrayEquals(new boolean[] {true, false, true}, ir.selectedAxes()),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity().contains("VARIANCE")),
                () -> assertTrue(ir.encodedKernelIr().familyIdentity().contains(
                        ":axes=[2, 0]:selected=[true, false, true]:keep=false:correction=1:")));
    }

    @Test void rejectsMismatchedMembershipRankPassCorrectionAndAccess() {
        var read = access(CpuAccessPlan.AccessKind.READ, 2);
        var write = access(CpuAccessPlan.AccessKind.WRITE, 1);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuAdvancedReductionIr(
                        CpuAdvancedReductionIr.Kind.L1_NORM, DataType.FLOAT64, new int[] {0},
                        new boolean[] {false, false}, false, 0, 1, 1, 2, 7, 64, read, write)),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuAdvancedReductionIr(
                        CpuAdvancedReductionIr.Kind.L2_NORM, DataType.FLOAT64, new int[] {0},
                        new boolean[] {true, false}, false, 1, 1, 1, 2, 0, 0, read, write)));
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.GENERAL_ODOMETER, rank,
                java.util.stream.IntStream.range(0, rank)
                        .mapToObj(i -> CpuAccessPlan.AxisRole.STRIDED).toList(), 0);
    }
}
