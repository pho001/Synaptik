package backend.kernels.cpu.layout;

import backend.kernels.cpu.CpuKernelContext;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorRemap;

import java.util.List;

final class LayoutExecutor {
    private LayoutExecutor() {
    }

    static void alias(List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.getFirst());
    }

    static void reshapeLike(List<Tensor> inputs, Tensor node) {
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
        TensorLayoutTransform.copyLinearized(src, node);
    }

    static void contiguous(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        TensorRemap.apply(inputs.getFirst(), node, context.contiguousMaterializeThreshold());
    }
}
