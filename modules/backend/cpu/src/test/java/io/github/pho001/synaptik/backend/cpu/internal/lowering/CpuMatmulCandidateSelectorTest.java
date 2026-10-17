package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
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
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 63, 48, 1, 1, 8, false), ComputePreference.VECTOR_IF_ELIGIBLE);
        var strided = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 2, 1, 8, false), ComputePreference.VECTOR_IF_ELIGIBLE);
        var terminal = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 1, 1, 8, true), ComputePreference.VECTOR_IF_ELIGIBLE);
        assertAll(() -> assertFalse(mixed.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(strided.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(terminal.candidates().contains(Realization.DIRECT_N_VECTOR)));
    }

    @Test void scalarPreferenceRetainsEligibleVectorCandidatesButSelectsScalarThresholdForm() {
        for (var type : new DataType[] {DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64}) {
            var direct = selector.select(new CpuMatmulCandidateSelector.Facts(type, type, type,
                    1, 2, 63, 128, 1, 1, 8, false), ComputePreference.SCALAR);
            var tiled = selector.select(new CpuMatmulCandidateSelector.Facts(type, type, type,
                    1, 32, 127, 256, 1, 1, 8, false), ComputePreference.SCALAR);
            assertAll(() -> assertEquals(Realization.DIRECT_SCALAR, direct.selected(), type.toString()),
                    () -> assertTrue(direct.candidates().contains(Realization.DIRECT_N_VECTOR)),
                    () -> assertEquals(Realization.TILED_SCALAR_2X2, tiled.selected(), type.toString()),
                    () -> assertTrue(tiled.candidates().contains(Realization.TILED_N_VECTOR_2X2)));
        }
    }

    @Test void allFourTypesRejectNonUnitStrideAndTerminalVectorCandidates() {
        for (var type : new DataType[] {DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64}) {
            for (boolean terminal : new boolean[] {false, true}) {
                long stride = terminal ? 1 : 2;
                var selected = selector.select(new CpuMatmulCandidateSelector.Facts(type, type,
                        type, 1, 32, 127, 256, stride, 1, 8, terminal),
                        ComputePreference.VECTOR_IF_ELIGIBLE);
                assertEquals(Realization.TILED_SCALAR_2X2, selected.selected(), type.toString());
                assertFalse(selected.candidates().contains(Realization.DIRECT_N_VECTOR));
                assertFalse(selected.candidates().contains(Realization.TILED_N_VECTOR_2X2));
            }
        }
    }

    private CpuMatmulCandidateSelector.Selection select(long m, long k, long n, boolean vector) {
        long stride = vector ? 1 : 2;
        return selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, m, k, n, stride, stride, 8, false), ComputePreference.VECTOR_IF_ELIGIBLE);
    }
}
