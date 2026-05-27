package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.reduceAll;

public final class CpuReduceAllKernel extends StorageAwareBoolReductionKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.REDUCE_ALL;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof reduceAll reduction)) {
            throw new IllegalArgumentException("CpuReduceAllKernel requires reduceAll operation");
        }
        return reduction.getDimension();
    }

    @Override
    protected boolean isAll() {
        return true;
    }
}
