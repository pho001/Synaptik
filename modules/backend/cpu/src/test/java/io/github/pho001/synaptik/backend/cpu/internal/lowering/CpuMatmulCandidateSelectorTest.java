package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue.Terminal;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

final class CpuMatmulCandidateSelectorTest {
    private final CpuMatmulCandidateSelector selector = new CpuMatmulCandidateSelector();

    @Test void onlyFloatingTanhExtendsVectorTerminalEligibility() {
        for (var type : DataType.values()) for (var terminal : Terminal.values()) {
            boolean eligible = (type == DataType.FLOAT32 || type == DataType.FLOAT64)
                    && terminal == Terminal.TANH;
            if (terminal == Terminal.NONE) continue;
            var facts = new CpuMatmulCandidateSelector.Facts(type, type, type,
                    1, 32, 127, 256, 1, 1, 8, terminal, 1);
            var selected = selector.select(facts, ComputePreference.VECTOR_IF_ELIGIBLE);
            assertEquals(eligible, selected.candidates().contains(Realization.DIRECT_N_VECTOR),
                    type + " " + terminal);
            assertEquals(eligible, selected.candidates().contains(Realization.TILED_N_VECTOR_2X2));
        }
        for (long stride : new long[] {0, 2}) {
            var facts = new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32, DataType.FLOAT32,
                    DataType.FLOAT32, 1, 32, 127, 256, 1, 1, 8, Terminal.TANH, stride);
            assertFalse(selector.select(facts, ComputePreference.VECTOR_IF_ELIGIBLE)
                    .candidates().contains(Realization.DIRECT_N_VECTOR));
            var resultStride = new CpuMatmulCandidateSelector.Facts(DataType.FLOAT64, DataType.FLOAT64,
                    DataType.FLOAT64, 1, 32, 127, 256, 1, stride, 8, Terminal.TANH, 1);
            assertFalse(selector.select(resultStride, ComputePreference.VECTOR_IF_ELIGIBLE)
                    .candidates().contains(Realization.DIRECT_N_VECTOR));
        }
    }

    @Test void tinyLegalTilesRemainCandidatesWithoutChangingProductionThresholds() {
        var tiny = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 2, 1, 16, 1, 1, 8, Terminal.NONE, 1),
                ComputePreference.SCALAR);
        assertEquals(java.util.List.of(Realization.DIRECT_SCALAR, Realization.DIRECT_N_VECTOR,
                Realization.TILED_SCALAR_2X2, Realization.TILED_N_VECTOR_2X2), tiny.candidates());
        assertEquals(Realization.DIRECT_SCALAR, tiny.selected());
    }
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
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 63, 48, 1, 1, 8, Terminal.NONE, 1), ComputePreference.VECTOR_IF_ELIGIBLE);
        var strided = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 2, 1, 8, Terminal.NONE, 1), ComputePreference.VECTOR_IF_ELIGIBLE);
        var terminal = selector.select(new CpuMatmulCandidateSelector.Facts(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, 1, 32, 127, 256, 1, 1, 8, Terminal.RELU, 1), ComputePreference.VECTOR_IF_ELIGIBLE);
        assertAll(() -> assertFalse(mixed.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(strided.candidates().contains(Realization.DIRECT_N_VECTOR)),
                () -> assertFalse(terminal.candidates().contains(Realization.DIRECT_N_VECTOR)));
    }

    @Test void scalarPreferenceRetainsEligibleVectorCandidatesButSelectsScalarThresholdForm() {
        for (var type : new DataType[] {DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64}) {
            var direct = selector.select(new CpuMatmulCandidateSelector.Facts(type, type, type,
                    1, 2, 63, 128, 1, 1, 8, Terminal.NONE, 1), ComputePreference.SCALAR);
            var tiled = selector.select(new CpuMatmulCandidateSelector.Facts(type, type, type,
                    1, 32, 127, 256, 1, 1, 8, Terminal.NONE, 1), ComputePreference.SCALAR);
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
                        type, 1, 32, 127, 256, stride, 1, 8, terminal ? Terminal.RELU : Terminal.NONE, 1),
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
                DataType.FLOAT32, DataType.FLOAT32, 1, m, k, n, stride, stride, 8, Terminal.NONE, 1), ComputePreference.VECTOR_IF_ELIGIBLE);
    }
}
