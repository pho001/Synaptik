package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionWeights;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class CpuScaledDotProductAttentionWeightsKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node, context);
        System.arraycopy(cached.getFloat64Data(), 0, node.getFloat64Data(), 0, node.getFlatDataSize());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node, context);
        System.arraycopy(cached.getFloat32Data(), 0, node.getFloat32Data(), 0, node.getFlatDataSize());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor cached = requireCachedWeights(op, inputs, node, context);
        if (cached.getDataType() == tensor.DataType.BFLOAT16) {
            System.arraycopy(cached.getBFloat16Data(), 0, node.getBFloat16Data(), 0, node.getFlatDataSize());
            return;
        }
        if (cached.getDataType() == tensor.DataType.FLOAT32) {
            float[] src = cached.getFloat32Data();
            short[] dst = node.getBFloat16Data();
            for (int i = 0; i < dst.length; i++) {
                dst[i] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(src[i]);
            }
            return;
        }
        throw new IllegalStateException("Unsupported cached weights dtype for BF16 export: " + cached.getDataType());
    }

    private static Tensor requireCachedWeights(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof scaledDotProductAttentionWeights)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionWeightsKernel requires scaledDotProductAttentionWeights operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("scaledDotProductAttentionWeights expects exactly one input");
        }
        Tensor attentionOut = inputs.getFirst();
        ScaledDotProductAttentionRuntimeCache runtimeCache =
                context == null ? null : context.runtimeStateFor(attentionOut, ScaledDotProductAttentionRuntimeCache.class);
        if (runtimeCache == null) {
            throw new IllegalStateException("scaledDotProductAttentionWeights requires cached forward weights on the attention output tensor");
        }
        Tensor weights = runtimeCache.weights();
        if (weights.getDataType() != node.getDataType()
                && !(weights.getDataType() == tensor.DataType.FLOAT32 && node.getDataType() == tensor.DataType.BFLOAT16)) {
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
