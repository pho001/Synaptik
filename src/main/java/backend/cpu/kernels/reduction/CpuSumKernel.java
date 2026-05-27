package backend.cpu.kernels.reduction;

import operations.Operation;
import operations.reduction.sum;
import tensor.Tensor;

import java.util.List;

public class CpuSumKernel extends StorageAwareSumLikeReductionKernel {
    @Override
    protected SumLikeReduction reduction() {
        return SumLikeReduction.SUM;
    }

    @Override
    protected int dimension(Operation op) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        return reduction.getDimension();
    }

    static Tensor requireSingleInput(List<Tensor> inputs, String label) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException(label + " expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
