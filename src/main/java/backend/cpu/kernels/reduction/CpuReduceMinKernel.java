package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.reduceMin;

public final class CpuReduceMinKernel extends StorageAwareMinMaxReductionKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.REDUCE_MIN;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof reduceMin reduction)) {
            throw new IllegalArgumentException("CpuReduceMinKernel requires reduceMin operation");
        }
        return reduction.getDimension();
    }

    @Override
    protected boolean isMax() {
        return false;
    }
}
