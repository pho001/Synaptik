package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorLayoutTransform;

import java.util.List;

public class CpuReshapeLikeKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forwardF64(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardAny(inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardAny(inputs, node);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardAny(inputs, node);
    }

    private static void forwardAny(List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        Tensor src = inputs.getFirst();
        if (src.getFlatDataSize() != node.getFlatDataSize()) {
            throw new IllegalArgumentException("Layout transform requires same number of elements.");
        }
        if (src.isContiguous()) {
            node.aliasRuntimeFrom(src);
            return;
        }
        TensorLayoutTransform.copyLinearized(src, node);
    }
}

