package backend.kernels.cpu;

import backend.kernels.cpu.reduction.SumLoops;
import backend.kernels.cpu.reduction.MeanSupport;
import operations.Operation;
import operations.mean;
import tensor.Tensor;

import java.util.List;

public final class CpuMeanKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof mean reduction)) {
            throw new IllegalArgumentException("CpuMeanKernel requires mean operation");
        }
        Tensor input = requireSingleInput(inputs);
        SumLoops.execute(input, node, reduction.getDimension(), context);
        MeanSupport.divideF64(node, divisor(input, reduction.getDimension()));
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof mean reduction)) {
            throw new IllegalArgumentException("CpuMeanKernel requires mean operation");
        }
        Tensor input = requireSingleInput(inputs);
        SumLoops.executeF32(input, node, reduction.getDimension(), context);
        MeanSupport.divideF32(node, divisor(input, reduction.getDimension()));
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof mean reduction)) {
            throw new IllegalArgumentException("CpuMeanKernel requires mean operation");
        }
        Tensor input = requireSingleInput(inputs);
        SumLoops.executeBF16(input, node, reduction.getDimension(), context);
        MeanSupport.divideBF16(node, divisor(input, reduction.getDimension()));
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Mean expects exactly one input tensor");
        }
        return inputs.getFirst();
    }

    private static int divisor(Tensor input, int dimension) {
        return dimension == -1 ? input.getFlatDataSize() : input.getShapeUnsafe()[dimension];
    }
}
