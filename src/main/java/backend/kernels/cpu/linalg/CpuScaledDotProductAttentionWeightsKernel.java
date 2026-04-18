package backend.kernels.cpu.linalg;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionWeights;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class CpuScaledDotProductAttentionWeightsKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node);
        System.arraycopy(cached.getFloat64Data(), 0, node.getFloat64Data(), 0, node.getFlatDataSize());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node);
        System.arraycopy(cached.getFloat32Data(), 0, node.getFloat32Data(), 0, node.getFlatDataSize());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node);
        System.arraycopy(cached.getBFloat16Data(), 0, node.getBFloat16Data(), 0, node.getFlatDataSize());
    }

    private static Tensor requireCachedWeights(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof scaledDotProductAttentionWeights)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionWeightsKernel requires scaledDotProductAttentionWeights operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("scaledDotProductAttentionWeights expects exactly one input");
        }
        Tensor attentionOut = inputs.getFirst();
        Object cached = attentionOut.getRuntimeCache();
        if (!(cached instanceof ScaledDotProductAttentionRuntimeCache runtimeCache)) {
            throw new IllegalStateException("scaledDotProductAttentionWeights requires cached forward weights on the attention output tensor");
        }
        Tensor weights = runtimeCache.weights();
        if (weights.getDataType() != node.getDataType()) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache dtype mismatch: cache="
                    + weights.getDataType() + ", node=" + node.getDataType());
        }
        if (!Arrays.equals(weights.getShapeUnsafe(), node.getShapeUnsafe())) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache shape mismatch: cache="
                    + Arrays.toString(weights.getShapeUnsafe()) + ", node=" + Arrays.toString(node.getShapeUnsafe()));
        }
        return weights;
    }
}
