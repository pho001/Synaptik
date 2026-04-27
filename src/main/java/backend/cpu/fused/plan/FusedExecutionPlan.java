package backend.cpu.fused.plan;

import backend.kernels.cpu.ResolvedCpuComputeContract;

import java.util.Objects;

public record FusedExecutionPlan(
        FusedOperation descriptor,
        ResolvedCpuComputeContract computeContract,
        int outputLength,
        int cpuVectorMinSize,
        int asmVectorWidth
) {
    public FusedExecutionPlan {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        Objects.requireNonNull(computeContract, "computeContract cannot be null");
        outputLength = Math.max(0, outputLength);
        cpuVectorMinSize = Math.max(1, cpuVectorMinSize);
        asmVectorWidth = Math.max(1, asmVectorWidth);
    }
}
