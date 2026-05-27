package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.reduceAny;

public final class CpuReduceAnyKernel extends StorageAwareBoolReductionKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.REDUCE_ANY;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof reduceAny reduction)) {
            throw new IllegalArgumentException("CpuReduceAnyKernel requires reduceAny operation");
        }
        return reduction.getDimension();
    }

    @Override
    protected boolean isAll() {
        return false;
    }
}
