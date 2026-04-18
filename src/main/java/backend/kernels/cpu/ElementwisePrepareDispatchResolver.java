package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.Objects;

public final class ElementwisePrepareDispatchResolver {
    private ElementwisePrepareDispatchResolver() {
    }

    public static ResolvedDispatchHints resolve(
            Operation op,
            Tensor node,
            ResolvedCpuComputeContract contract,
            CpuExecutionPlanner planner
    ) {
        Objects.requireNonNull(op, "op cannot be null");
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(contract, "contract cannot be null");
        Objects.requireNonNull(planner, "planner cannot be null");

        if (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1, false);
        }

        int totalLength = Math.max(0, node.getFlatDataSize());
        CpuKernelCostClass costClass = planner.dispatchCostClass(op);
        int targetChunks = planner.targetChunksPerWorker(costClass);
        int vectorWidth = planner.preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1 && totalLength >= planner.elementwiseVectorMinSize(op);
        boolean parallelAllowed = totalLength >= planner.elementwiseParallelMinSize(op);

        CpuExecutionMode mode;
        if (parallelAllowed) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        return new ResolvedDispatchHints(
                totalLength,
                mode,
                planner.computeChunkSize(totalLength, 1, targetChunks, planner.minScalarChunkSize()),
                planner.computeChunkSize(totalLength, vectorWidth, targetChunks, planner.minVectorChunkSize()),
                vectorWidth,
                planner.plannedWorkers(),
                (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR)
                        && planner.shouldUseCommonPoolFor(costClass, totalLength)
        );
    }
}
