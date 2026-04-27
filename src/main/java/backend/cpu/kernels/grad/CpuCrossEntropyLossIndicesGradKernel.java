package backend.cpu.kernels.grad;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.loss.crossEntropyLossIndicesGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuCrossEntropyLossIndicesGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndicesGrad grad = require(op);
        Tensor[] triple = requireTriple(inputs);
        CrossEntropyLossIndicesGradExecutor.executeF64(grad, triple[0], triple[1], triple[2], node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndicesGrad grad = require(op);
        Tensor[] triple = requireTriple(inputs);
        CrossEntropyLossIndicesGradExecutor.executeF32(grad, triple[0], triple[1], triple[2], node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndicesGrad grad = require(op);
        Tensor[] triple = requireTriple(inputs);
        CrossEntropyLossIndicesGradExecutor.executeBF16(grad, triple[0], triple[1], triple[2], node, context);
    }

    private static crossEntropyLossIndicesGrad require(Operation op) {
        if (!(op instanceof crossEntropyLossIndicesGrad grad)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossIndicesGradKernel requires crossEntropyLossIndicesGrad operation");
        }
        return grad;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("Cross entropy loss indices grad expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
