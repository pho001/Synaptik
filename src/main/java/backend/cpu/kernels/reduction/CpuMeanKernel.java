package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.mean;

public final class CpuMeanKernel extends StorageAwareSumLikeReductionKernel {
    @Override
    protected SumLikeReduction reduction() {
        return SumLikeReduction.MEAN;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof mean reduction)) {
            throw new IllegalArgumentException("CpuMeanKernel requires mean operation");
        }
        return reduction.getDimension();
    }
}
