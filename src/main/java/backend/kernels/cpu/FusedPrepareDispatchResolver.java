package backend.kernels.cpu;

import operations.fused.FusedOperation;
import tensor.Tensor;

import java.util.Objects;

public final class FusedPrepareDispatchResolver {
    private FusedPrepareDispatchResolver() {
    }

    public static PreparedFusedDispatch resolve(
            FusedOperation fused,
            Tensor node,
            ResolvedCpuComputeContract contract,
            CpuExecutionPlanner planner
    ) {
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(contract, "contract cannot be null");
        Objects.requireNonNull(planner, "planner cannot be null");

        int totalLength = Math.max(0, node.getFlatDataSize());
        CpuKernelCostClass costClass = fusedCostClass(fused);
        int targetChunks = planner.targetChunksPerWorker(costClass);
        int cpuVectorMinSize = planner.fusedDirectVectorMinSize(fused);

        if (planner.shouldForceSerialScalarDispatch(fused)) {
            return new PreparedFusedDispatch(
                    new ResolvedDispatchHints(
                            totalLength,
                            CpuExecutionMode.SCALAR,
                            planner.computeChunkSize(totalLength, 1, targetChunks, planner.minScalarChunkSize()),
                            planner.computeChunkSize(totalLength, 1, targetChunks, planner.minVectorChunkSize()),
                            1,
                            1,
                            false
                    ),
                    cpuVectorMinSize,
                    1
            );
        }

        int asmVectorWidth = planner.resolvedFusedAsmVectorWidth(contract, fused);
        boolean vectorAllowed = asmVectorWidth > 1 && totalLength >= cpuVectorMinSize;
        boolean parallelAllowed = totalLength >= planner.fusedParallelMinSize(fused);

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
                        planner.computeChunkSize(totalLength, 1, targetChunks, planner.minScalarChunkSize()),
                        planner.computeChunkSize(totalLength, asmVectorWidth, targetChunks, planner.minVectorChunkSize()),
                        asmVectorWidth,
                        planner.plannedWorkers(),
                        (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR)
                                && planner.shouldUseCommonPoolFor(costClass, totalLength)
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
