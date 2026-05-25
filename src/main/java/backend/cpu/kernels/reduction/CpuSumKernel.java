package backend.cpu.kernels.reduction;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.reduction.sum;
import tensor.Tensor;

import java.util.List;

public class CpuSumKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sum reduction = require(op);
        Tensor input = requireSingleInput(inputs, "Sum");
        SumLikeReductionExecutor.executeF64(SumLikeReduction.SUM, input, node, reduction.getDimension(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sum reduction = require(op);
        Tensor input = requireSingleInput(inputs, "Sum");
        SumLikeReductionExecutor.executeF32(SumLikeReduction.SUM, input, node, reduction.getDimension(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sum reduction = require(op);
        Tensor input = requireSingleInput(inputs, "Sum");
        SumLikeReductionExecutor.executeBF16(SumLikeReduction.SUM, input, node, reduction.getDimension(), context);
    }

    private static sum require(Operation op) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        return reduction;
    }

    static Tensor requireSingleInput(List<Tensor> inputs, String label) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException(label + " expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
