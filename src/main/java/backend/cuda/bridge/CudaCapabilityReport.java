package backend.cuda.bridge;

import java.util.List;
import java.util.Objects;

/**
 * Layered CUDA capability report used by parity and coverage diagnostics.
 */
public record CudaCapabilityReport(List<Entry> entries) {
    public CudaCapabilityReport {
        entries = entries == null ? List.of() : entries.stream()
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * Returns true only when CUDA graph execution capability is available.
     */
    public boolean graphExecutable() {
        return hasStatus(CudaCapabilityDimension.GRAPH_EXECUTION_ABI, CudaCapabilityDimensionStatus.AVAILABLE)
                && hasStatus(CudaCapabilityDimension.CONTEXT, CudaCapabilityDimensionStatus.AVAILABLE);
    }

    /**
     * Returns true only when native buffer binding execution capability is available.
     */
    public boolean bufferBindingExecutable() {
        return hasStatus(CudaCapabilityDimension.BUFFER_BINDING_ABI, CudaCapabilityDimensionStatus.AVAILABLE);
    }

    /**
     * Returns true only when layout ABI v2 is available.
     */
    public boolean layoutAbiV2Executable() {
        return hasStatus(CudaCapabilityDimension.LAYOUT_ABI_V2, CudaCapabilityDimensionStatus.AVAILABLE);
    }

    /**
     * Capability skips are never support evidence.
     */
    public boolean capabilitySkipCountsAsSupport() {
        return false;
    }

    public List<Entry> entriesFor(CudaCapabilityDimension dimension) {
        return entries.stream()
                .filter(entry -> entry.dimension() == dimension)
                .toList();
    }

    public boolean hasStatus(CudaCapabilityDimension dimension, CudaCapabilityDimensionStatus status) {
        return entries.stream()
                .anyMatch(entry -> entry.dimension() == dimension && entry.status() == status);
    }

    /**
     * One capability dimension status row.
     */
    public record Entry(CudaCapabilityDimension dimension, CudaCapabilityDimensionStatus status, String detail) {
        public Entry {
            Objects.requireNonNull(dimension, "dimension");
            Objects.requireNonNull(status, "status");
            detail = detail == null ? "" : detail.strip();
        }
    }
}
