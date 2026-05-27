package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorRemap;

import java.util.List;

public class CpuReshapeLikeKernel extends TypedCpuKernel implements CpuLayoutOutputStorageDeferredKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reshapeLike(op, inputs, node, context);
    }

    private static void reshapeLike(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (CpuLayoutNativeViewSupport.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        Tensor src = inputs.getFirst();
        if (src.getFlatDataSize() != node.getFlatDataSize()) {
            throw new IllegalArgumentException("Layout transform requires same number of elements.");
        }
        if (src.isContiguous()) {
            TensorInternalAccess.aliasRuntimeFrom(node, src);
            return;
        }
        TensorRemap.copyLinearized(src, node);
    }
}
