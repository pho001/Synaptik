package backend.cpu.fused.plan;

import backend.cpu.kernels.ResolvedCpuComputeContract;

import java.util.Objects;

/**
 * Runtime execution contract for a lowered fused CPU operation.
 *
 * @param descriptor fused operation descriptor and expression plan
 * @param computeContract resolved output shape, dtype, and layout contract
 * @param outputLength flattened number of output elements to compute
 * @param cpuVectorMinSize minimum range size before vector dispatch is considered
 * @param asmVectorWidth preferred ASM vector width in lanes
 */
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
