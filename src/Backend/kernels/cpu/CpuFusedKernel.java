package Backend.kernels.cpu;

import Operations.FusedCompiledOperation;
import Operations.FusedOperation;
import Operations.Operation;
import Tensor.Tensor;
import Graph.codegen.FusedVectorOps;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
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
            case PARALLEL -> runParallel(ranged, inputs, node, config, false, fused.getPrecisionMode());
            case PARALLEL_VECTOR -> runParallel(ranged, inputs, node, config, recommendVector, fused.getPrecisionMode());
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
            int precisionMode
    ) {
        int length = node.getFlatDataSize();
        int width = preferVector ? Math.max(1, FusedVectorOps.width(precisionMode)) : 1;
        int chunkSize = config.computeChunkSize(length, width);
        int chunks = (length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, length);
            if (preferVector) {
                ranged.applyRangeVector(inputs, node, start, end);
            } else {
                ranged.applyRangeScalar(inputs, node, start, end);
            }
        });
    }
}
