package Backend.kernels.cpu;

import Operations.FusedCompiledOperation;
import Operations.FusedOperation;
import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
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
        switch (mode) {
            case SCALAR -> ranged.applyRangeScalar(inputs, node, 0, length);
            case VECTOR -> ranged.applyRangeVector(inputs, node, 0, length);
            case PARALLEL -> runParallel(ranged, inputs, node, config, false);
            case PARALLEL_VECTOR -> runParallel(ranged, inputs, node, config, true);
        }
    }

    private static void runParallel(
            FusedCompiledOperation ranged,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionConfig config,
            boolean preferVector
    ) {
        int length = node.getFlatDataSize();
        int chunkSize = config.computeChunkSize(length, 1);
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
