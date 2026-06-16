package backend.cpu.prepare.elementwise;

import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.plan.CpuKernelCostClass;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.prepare.CpuPlanningPolicy;
import operations.Operation;
import tensor.Tensor;

public final class ElementwiseDispatchPlanner {
    private final CpuPlanningPolicy policy;

    public ElementwiseDispatchPlanner(CpuPlanningPolicy policy) {
        this.policy = policy;
    }

    public ResolvedDispatchHints resolve(Operation op, Tensor node, ResolvedCpuComputeContract contract) {
        if (op == null || node == null
                || op.arityClass() != Operation.OpArityClass.ELEMENT_WISE) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1, false);
        }

        return resolve(op, node.getFlatDataSize(), contract);
    }

    public ResolvedDispatchHints resolve(Operation op, long logicalElementCount, ResolvedCpuComputeContract contract) {
        if (op == null || op.arityClass() != Operation.OpArityClass.ELEMENT_WISE) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1, false);
        }

        int totalLength = (int) Math.min(Integer.MAX_VALUE, Math.max(0L, logicalElementCount));
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
