package backend.cpu.kernels.fused.plan;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuKernelCostClass;
import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.plan.CpuPlanningPolicy;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedVectorFallbackReason;
import backend.cpu.fused.plan.FusedVectorGuard;
import tensor.Tensor;

import java.util.Objects;

public final class FusedDispatchPlanner {
    private final CpuPlanningPolicy policy;

    public FusedDispatchPlanner(CpuPlanningPolicy policy) {
        this.policy = Objects.requireNonNull(policy, "policy cannot be null");
    }

    public PreparedFusedDispatch resolve(
            FusedOperation fused,
            Tensor node,
            ResolvedCpuComputeContract contract
    ) {
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(contract, "contract cannot be null");

        return resolve(fused, node.getFlatDataSize(), contract);
    }

    public PreparedFusedDispatch resolve(
            FusedOperation fused,
            long logicalElementCount,
            ResolvedCpuComputeContract contract
    ) {
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(contract, "contract cannot be null");

        int totalLength = (int) Math.min(Integer.MAX_VALUE, Math.max(0L, logicalElementCount));
        CpuKernelCostClass costClass = fusedCostClass(fused);
        int targetChunks = policy.targetChunksPerWorker(costClass);
        int cpuVectorMinSize = policy.fusedDirectVectorMinSize(fused);
        FusedVectorFallbackReason dispatchFallbackReason = FusedVectorGuard.dispatchFallbackReason(fused);

        if (dispatchFallbackReason.requiresSerialScalarDispatch()) {
            return new PreparedFusedDispatch(
                    new ResolvedDispatchHints(
                            totalLength,
                            CpuExecutionMode.SCALAR,
                            policy.computeChunkSize(totalLength, 1, targetChunks, policy.minScalarChunkSize()),
                            policy.computeChunkSize(totalLength, 1, targetChunks, policy.minVectorChunkSize()),
                            1,
                            1,
                            false
                    ),
                    cpuVectorMinSize,
                    1,
                    dispatchFallbackReason
            );
        }

        int asmVectorWidth = policy.resolvedFusedAsmVectorWidth(contract, fused);
        boolean vectorAllowed = asmVectorWidth > 1 && totalLength >= cpuVectorMinSize;
        boolean parallelAllowed = totalLength >= policy.fusedParallelMinSize(fused);

        CpuExecutionMode mode;
        if (parallelAllowed) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        return new PreparedFusedDispatch(
                new ResolvedDispatchHints(
                        totalLength,
                        mode,
                        policy.computeChunkSize(totalLength, 1, targetChunks, policy.minScalarChunkSize()),
                        policy.computeChunkSize(totalLength, asmVectorWidth, targetChunks, policy.minVectorChunkSize()),
                        asmVectorWidth,
                        policy.plannedWorkers(),
                        (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR)
                                && policy.shouldUseCommonPoolFor(costClass, totalLength)
                ),
                cpuVectorMinSize,
                asmVectorWidth,
                FusedVectorGuard.preparedFallbackReason(contract, fused, totalLength, cpuVectorMinSize, asmVectorWidth)
        );
    }

    private static CpuKernelCostClass fusedCostClass(FusedOperation fused) {
        return fused.isLowCostHint() ? CpuKernelCostClass.LOW : CpuKernelCostClass.MEDIUM;
    }
}
