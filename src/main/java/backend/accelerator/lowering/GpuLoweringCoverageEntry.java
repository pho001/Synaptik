package backend.accelerator.lowering;

import backend.ComputeBackend;
import operations.Operation;

import java.util.Objects;

/**
 * Single source-level row in the GPU lowering coverage matrix.
 */
public record GpuLoweringCoverageEntry(
        ComputeBackend backend,
        Operation.OpType opType,
        GpuLoweringOperationFamily family,
        GpuLoweringCoverageStatus status,
        GpuLoweringUnsupportedReason reason,
        String note
) {
    public GpuLoweringCoverageEntry {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(opType, "opType");
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(reason, "reason");
        note = note == null ? "" : note.strip();
        if (status == GpuLoweringCoverageStatus.SUPPORTED && reason != GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("supported coverage rows must use SUPPORTED reason");
        }
        if (status != GpuLoweringCoverageStatus.SUPPORTED && reason == GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("non-supported coverage rows must carry a concrete reason");
        }
    }
}
