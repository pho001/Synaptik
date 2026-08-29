package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

final class CpuMatmulCandidateSelectorTest {
    private final CpuMatmulCandidateSelector selector = new CpuMatmulCandidateSelector();

    @Test void appliesExactBoundedThresholdsAndAlwaysRetainsScalar() {
        assertEquals(Realization.DIRECT_SCALAR, select(1, 4096, 1, true).selected());
        assertEquals(Realization.DIRECT_N_VECTOR, select(2, 16, 128, true).selected());
        assertEquals(Realization.TILED_SCALAR_2X2, select(32, 16, 32, false).selected());
        var tiled = select(32, 8, 256, true);
        assertEquals(Realization.TILED_N_VECTOR_2X2, tiled.selected());
        assertEquals(Realization.DIRECT_SCALAR, tiled.candidates().getFirst());
        assertTrue(tiled.candidates().size() <= 4);
    }

    @Test void excludesVectorForMixedTypesNonUnitStrideAndTerminal() {
        var mixed = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.BFLOAT16,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 63, 48, 1, 1, 8, false));
        var strided = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 2, 1, 8, false));
        var terminal = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 1, 1, 8, true));
        assertAll(() -> assertFalse(mixed.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(strided.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(terminal.candidates().contains(Realization.DIRECT_N_VECTOR)));
    }

    private CpuMatmulCandidateSelector.Selection select(long m, long k, long n, boolean vector) {
        long stride = vector ? 1 : 2;
        return selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, m, k, n, stride, stride, 8, false));
    }
}
