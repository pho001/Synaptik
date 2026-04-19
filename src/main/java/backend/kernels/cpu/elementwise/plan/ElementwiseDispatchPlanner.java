package backend.kernels.cpu.elementwise.plan;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelCostClass;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.plan.CpuPlanningPolicy;
import operations.Operation;
import tensor.Tensor;

public final class ElementwiseDispatchPlanner {
    private final CpuPlanningPolicy policy;

    public ElementwiseDispatchPlanner(CpuPlanningPolicy policy) {
        this.policy = policy;
    }

    public ResolvedDispatchHints resolve(Operation op, Tensor node, ResolvedCpuComputeContract contract) {
        if (op == null || node == null
                || op.opType().category() != Operation.OpArityClass.ELEMENT_WISE) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1, false);
        }

        int totalLength = Math.max(0, node.getFlatDataSize());
        CpuKernelCostClass costClass = policy.dispatchCostClass(op);
        int vectorWidth = policy.preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1 && totalLength >= policy.elementwiseVectorMinSize(op);

        CpuExecutionMode mode;
        if (totalLength >= policy.elementwiseParallelMinSize(op)) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        return new ResolvedDispatchHints(
                totalLength,
                mode,
                policy.computeChunkSize(totalLength, 1, policy.targetChunksPerWorker(costClass), policy.minScalarChunkSize()),
                policy.computeChunkSize(totalLength, vectorWidth, policy.targetChunksPerWorker(costClass), policy.minVectorChunkSize()),
                vectorWidth,
                policy.plannedWorkers(),
                (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR)
                        && policy.shouldUseCommonPoolFor(costClass, totalLength)
        );
    }
}
