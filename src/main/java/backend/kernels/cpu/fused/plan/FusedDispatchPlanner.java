package backend.kernels.cpu.fused.plan;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelCostClass;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.plan.CpuPlanningPolicy;
import backend.cpu.fused.plan.FusedOperation;
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

        int totalLength = Math.max(0, node.getFlatDataSize());
        CpuKernelCostClass costClass = fusedCostClass(fused);
        int targetChunks = policy.targetChunksPerWorker(costClass);
        int cpuVectorMinSize = policy.fusedDirectVectorMinSize(fused);

        if (policy.shouldForceSerialScalarDispatch(fused)) {
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
                    1
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
                asmVectorWidth
        );
    }

    private static CpuKernelCostClass fusedCostClass(FusedOperation fused) {
        return fused.isLowCostHint() && fused.getDispatchScale() == 1
                ? CpuKernelCostClass.LOW
                : CpuKernelCostClass.MEDIUM;
    }
}
