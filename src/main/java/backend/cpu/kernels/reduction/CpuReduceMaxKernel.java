package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.reduceMax;

public final class CpuReduceMaxKernel extends StorageAwareMinMaxReductionKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.REDUCE_MAX;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof reduceMax reduction)) {
            throw new IllegalArgumentException("CpuReduceMaxKernel requires reduceMax operation");
        }
        return reduction.getDimension();
    }

    @Override
    protected boolean isMax() {
        return true;
    }
}
