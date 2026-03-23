package Backend.kernels.cpu;

import Operations.FusedCompiledOperation;
import Operations.FusedOperation;
import Operations.Operation;
import Tensor.Tensor;
import Graph.codegen.FusedVectorOps;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
    @Override
    public CpuKernelCostClass costClass(Operation op) {
        if (op instanceof FusedOperation fused) {
            return fused.isLowCostHint() ? CpuKernelCostClass.LOW : CpuKernelCostClass.MEDIUM;
        }
        return CpuKernel.super.costClass(op);
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forwardF64(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        if (!(op instanceof FusedOperation fused)) {
            op.apply(inputs, node);
            return;
        }

        Operation compiled = fused.getCompiledInstance();
        if (!(compiled instanceof FusedCompiledOperation ranged)) {
            compiled.apply(inputs, node);
            return;
        }

        int length = node.getFlatDataSize();
        CpuExecutionMode mode = config.modeFor(op, node);
        CpuKernelCostClass costClass = fused.isLowCostHint() ? CpuKernelCostClass.LOW : CpuKernelCostClass.MEDIUM;
        String schedulerKey = fused.getSchedulerSignature();
        boolean recommendVector = true;
        if (fused != null) {
            recommendVector = FusedVectorOps.isRecommended(fused.getPrecisionMode());
        }
        switch (mode) {
            case SCALAR -> ranged.applyRangeScalar(inputs, node, 0, length);
            case VECTOR -> {
                if (recommendVector) {
                    ranged.applyRangeVector(inputs, node, 0, length);
                } else {
                    ranged.applyRangeScalar(inputs, node, 0, length);
                }
            }
            case PARALLEL -> runParallel(ranged, inputs, node, config, false, fused.getPrecisionMode(), costClass, schedulerKey);
            case PARALLEL_VECTOR -> runParallel(ranged, inputs, node, config, recommendVector, fused.getPrecisionMode(), costClass, schedulerKey);
        }
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    private static void runParallel(
            FusedCompiledOperation ranged,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionConfig config,
            boolean preferVector,
            int precisionMode,
            CpuKernelCostClass costClass,
            String schedulerKey
    ) {
        int length = node.getFlatDataSize();
        int width = preferVector ? Math.max(1, FusedVectorOps.width(precisionMode)) : 1;
        int chunkSize = config.computeChunkSize(length, width);
        int chunks = (length + chunkSize - 1) / chunkSize;
        boolean useCommonPool = CpuSchedulerAdvisor.shouldUseCommonPool(costClass, schedulerKey, length, config);
        long t0 = System.nanoTime();
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, length);
            if (preferVector) {
                ranged.applyRangeVector(inputs, node, start, end);
            } else {
                ranged.applyRangeScalar(inputs, node, start, end);
            }
        }, useCommonPool);
        CpuSchedulerAdvisor.recordSample(schedulerKey, length, System.nanoTime() - t0);
    }
}
