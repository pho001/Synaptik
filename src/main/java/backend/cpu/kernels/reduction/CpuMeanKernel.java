package backend.cpu.kernels.reduction;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.reduction.mean;
import tensor.Tensor;

import java.util.List;

public final class CpuMeanKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mean reduction = require(op);
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "Mean");
        SumLikeReductionExecutor.executeF64(SumLikeReduction.MEAN, input, node, reduction.getDimension(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mean reduction = require(op);
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "Mean");
        SumLikeReductionExecutor.executeF32(SumLikeReduction.MEAN, input, node, reduction.getDimension(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mean reduction = require(op);
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "Mean");
        SumLikeReductionExecutor.executeBF16(SumLikeReduction.MEAN, input, node, reduction.getDimension(), context);
    }

    private static mean require(Operation op) {
        if (!(op instanceof mean reduction)) {
            throw new IllegalArgumentException("CpuMeanKernel requires mean operation");
        }
        return reduction;
    }
}
