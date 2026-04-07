package graph.fused;

import backend.kernels.cpu.CpuComputeMode;
import operations.FusedOperation;

import java.util.Objects;

public record FusedExecutionPlan(
        FusedOperation descriptor,
        CpuComputeMode computeMode,
        int outputLength,
        int cpuVectorMinSize
) {
    public FusedExecutionPlan {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        Objects.requireNonNull(computeMode, "computeMode cannot be null");
        outputLength = Math.max(0, outputLength);
        cpuVectorMinSize = Math.max(1, cpuVectorMinSize);
    }
}
