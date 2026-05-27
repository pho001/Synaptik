package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionWeights;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.Arrays;
import java.util.List;

public final class CpuScaledDotProductAttentionWeightsKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        Tensor node = call.outputTensor();
        Tensor cached = requireCachedWeights(call.operation(), call.inputTensors(), node, call.context());
        switch (node.getDataType()) {
            case FLOAT64 -> System.arraycopy(TensorInternalAccess.float64Data(cached), 0,
                    TensorInternalAccess.float64Data(node), 0, node.getFlatDataSize());
            case FLOAT32 -> System.arraycopy(TensorInternalAccess.float32Data(cached), 0,
                    TensorInternalAccess.float32Data(node), 0, node.getFlatDataSize());
            case BFLOAT16 -> publishBF16(cached, node);
            case INT32, INT64, BOOL -> unsupported(node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static void publishBF16(Tensor cached, Tensor node) {
        if (cached.getDataType() == DataType.BFLOAT16) {
            System.arraycopy(TensorInternalAccess.bfloat16Data(cached), 0,
                    TensorInternalAccess.bfloat16Data(node), 0, node.getFlatDataSize());
            return;
        }
        if (cached.getDataType() == DataType.FLOAT32) {
            float[] src = TensorInternalAccess.float32Data(cached);
            short[] dst = TensorInternalAccess.bfloat16Data(node);
            for (int i = 0; i < dst.length; i++) {
                dst[i] = tensor.dtype.TensorDTypeOps.toBFloat16Bits(src[i]);
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
                && !(weights.getDataType() == DataType.FLOAT32 && node.getDataType() == DataType.BFLOAT16)) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache dtype mismatch: cache="
                    + weights.getDataType() + ", node=" + node.getDataType());
        }
        if (!Arrays.equals(weights.getShapeUnsafe(), node.getShapeUnsafe())) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache shape mismatch: cache="
                    + Arrays.toString(weights.getShapeUnsafe()) + ", node=" + Arrays.toString(node.getShapeUnsafe()));
        }
        return weights;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuScaledDotProductAttentionWeightsKernel does not support " + dtype);
    }
}
