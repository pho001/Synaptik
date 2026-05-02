package backend.accelerator.lowering;

import backend.ComputeBackend;

import java.util.Comparator;

/**
 * Builds derived backend parity reports from the checked-in GPU lowering matrix.
 */
public final class GpuBackendParityReporter {
    private static final Comparator<GpuLoweringCoverageEntry> ENTRY_ORDER = Comparator
            .comparing((GpuLoweringCoverageEntry entry) -> entry.family().name())
            .thenComparing(entry -> entry.opType().name());

    private GpuBackendParityReporter() {
    }

    /**
     * Compares CUDA coverage rows against every Metal coverage row.
     */
    public static GpuBackendParityReport cudaAgainstMetal() {
        return new GpuBackendParityReport(
                GpuLoweringCoverageMatrix.entriesFor(ComputeBackend.GPU_METAL).stream()
                        .sorted(ENTRY_ORDER)
                        .map(metal -> GpuBackendParityRow.from(
                                metal,
                                GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, metal.opType())
                        ))
                        .toList()
        );
    }
}
