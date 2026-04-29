package graph.execution.trace;

import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;

import java.util.Objects;

/**
 * Trace entry for a runtime point that needs CPU-readable tensor storage.
 *
 * <p>The trace is emitted by execution-state residency checks. A completed entry means a
 * device-to-CPU synchronization has already happened before the state transition was recorded. An
 * incomplete entry means execution discovered that CPU storage was required but no materializer was
 * available, so continuing would risk reading stale tensor arrays.</p>
 *
 * @param nodeId compiled node id whose value was requested on CPU
 * @param reason why CPU-readable storage was needed
 * @param materializedFrom backend that owned the current device value, or an empty string for CPU/no backend
 * @param sourceResidency residency before CPU materialization was requested
 * @param bytes logical payload bytes involved in the request
 * @param durationNs measured materialization duration, or zero when no materializer ran
 * @param completed whether CPU storage was synchronized successfully
 * @param detail diagnostic detail
 */
public record CpuMaterializationTrace(
        int nodeId,
        CpuMaterializationReason reason,
        String materializedFrom,
        StorageResidency sourceResidency,
        long bytes,
        long durationNs,
        boolean completed,
        String detail
) {
    public CpuMaterializationTrace {
        Objects.requireNonNull(reason, "reason cannot be null");
        Objects.requireNonNull(sourceResidency, "sourceResidency cannot be null");
        materializedFrom = materializedFrom == null ? "" : materializedFrom;
        bytes = Math.max(0L, bytes);
        durationNs = Math.max(0L, durationNs);
        detail = detail == null ? "" : detail;
    }
}
