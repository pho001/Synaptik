package backend.accelerator.lowering;

import backend.ComputeBackend;
import operations.Operation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuLoweringCoverageMatrixTest {
    private static final Set<GpuLoweringOperationFamily> REQUIRED_PHASE_ELEVEN_FAMILIES = EnumSet.of(
            GpuLoweringOperationFamily.MATMUL_LINEAR,
            GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
            GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
            GpuLoweringOperationFamily.SOFTMAX_LIKE,
            GpuLoweringOperationFamily.REDUCTION,
            GpuLoweringOperationFamily.NORMALIZATION,
            GpuLoweringOperationFamily.LOSS_ADJACENT,
            GpuLoweringOperationFamily.ATTENTION
    );

    @Test
    void matrixCoversRequiredPhaseElevenFamiliesForMetalAndCuda() {
        assertRequiredFamiliesCovered(ComputeBackend.GPU_METAL);
        assertRequiredFamiliesCovered(ComputeBackend.GPU_CUDA);
    }

    @Test
    void supportedEntriesHaveConcreteOperationTypes() {
        for (GpuLoweringCoverageEntry entry : GpuLoweringCoverageMatrix.entries()) {
            if (entry.status() == GpuLoweringCoverageStatus.SUPPORTED) {
                assertNotNull(entry.opType(), () -> "missing op type for supported entry " + entry);
                assertFalse(entry.opType() == Operation.OpType.UNKNOWN,
                        () -> "supported entry must not use UNKNOWN op type: " + entry);
                assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                        () -> "supported entry must use SUPPORTED reason: " + entry);
            }
        }
    }

    @Test
    void nonSupportedEntriesHaveStableReasons() {
        for (GpuLoweringCoverageEntry entry : GpuLoweringCoverageMatrix.entries()) {
            if (entry.status() == GpuLoweringCoverageStatus.FALLBACK
                    || entry.status() == GpuLoweringCoverageStatus.UNSUPPORTED) {
                assertNotNull(entry.reason(), () -> "missing reason for non-supported entry " + entry);
                assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED,
                        () -> "non-supported entry must not use SUPPORTED reason: " + entry);
            } else {
                assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status(),
                        () -> "unknown coverage status: " + entry);
            }
        }
    }

    @Test
    void docsMatrixListsRequiredStatusesAndReasons() throws IOException {
        String docs = Files.readString(Path.of("docs/gpu-lowering-coverage.md"));

        assertTrue(docs.contains("GPU Lowering Coverage Matrix"));
        assertTrue(docs.contains("GPULOWER-01"));
        assertTrue(docs.contains("UNSUPPORTED_OPERATION"));
        assertTrue(docs.contains("UNSUPPORTED_DTYPE"));
        assertTrue(docs.contains("UNSUPPORTED_LAYOUT"));
        assertTrue(docs.contains("DEFERRED_FUSED_REGION"));
        assertTrue(docs.contains("supported"));
        assertTrue(docs.contains("fallback"));
        assertTrue(docs.contains("unsupported"));
    }

    @Test
    void compoundFusedRowsExposeStablePhaseTwelveReasons() {
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.FUSED);

            assertFalse(entry.status() == GpuLoweringCoverageStatus.SUPPORTED);
            assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED);
            assertEquals(GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED, entry.reason());
            assertTrue(entry.note().contains("CPU Operation.OpType.FUSED remains CPU-only for Phase 12"));
        }
    }

    private static void assertRequiredFamiliesCovered(ComputeBackend backend) {
        List<GpuLoweringCoverageEntry> entries = GpuLoweringCoverageMatrix.entriesFor(backend);
        Set<GpuLoweringOperationFamily> coveredFamilies = entries.stream()
                .map(GpuLoweringCoverageEntry::family)
                .collect(Collectors.toCollection(() -> EnumSet.noneOf(GpuLoweringOperationFamily.class)));

        assertTrue(coveredFamilies.containsAll(REQUIRED_PHASE_ELEVEN_FAMILIES),
                () -> backend + " missing required families: " + missing(coveredFamilies));
        assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.SUPPORTED),
                () -> backend + " must have supported coverage rows");
        assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.FALLBACK),
                () -> backend + " must have fallback coverage rows");
        assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.UNSUPPORTED),
                () -> backend + " must have unsupported coverage rows");
    }

    private static Set<GpuLoweringOperationFamily> missing(Set<GpuLoweringOperationFamily> coveredFamilies) {
        Set<GpuLoweringOperationFamily> missing = EnumSet.copyOf(REQUIRED_PHASE_ELEVEN_FAMILIES);
        missing.removeAll(coveredFamilies);
        return missing;
    }
}
