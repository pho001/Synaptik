package backend.kernels.cpu;

import backend.kernels.cpu.reduction.SoftmaxExecutor;
import operations.Operation;
import operations.softmax;
import tensor.Tensor;

import java.util.List;

public final class CpuSoftmaxKernel implements CpuKernel {
    private static final SoftmaxExecutor EXECUTOR = new SoftmaxExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof softmax reduction)) {
            throw new IllegalArgumentException("CpuSoftmaxKernel requires softmax operation");
        }
        EXECUTOR.execute(reduction, requireSingleInput(inputs), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof softmax reduction)) {
            throw new IllegalArgumentException("CpuSoftmaxKernel requires softmax operation");
        }
        EXECUTOR.executeF32(reduction, requireSingleInput(inputs), node, context);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof softmax reduction)) {
            throw new IllegalArgumentException("CpuSoftmaxKernel requires softmax operation");
        }
        EXECUTOR.executeF16(reduction, requireSingleInput(inputs), node, context);
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Softmax expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
