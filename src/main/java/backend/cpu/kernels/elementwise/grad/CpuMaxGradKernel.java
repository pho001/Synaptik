package backend.cpu.kernels.elementwise.grad;

import backend.cpu.execution.CpuKernelContext;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.elementwise.binary.maxGrad;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxGradKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        maxGrad gradOp = (maxGrad) op;
        runF64(
                TensorInternalAccess.float64Data(inputs.get(0)), inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.float64Data(inputs.get(1)), inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.float64Data(inputs.get(2)), inputs.get(2).getStorageOffsetUnsafe(),
                TensorInternalAccess.float64Data(node), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        maxGrad gradOp = (maxGrad) op;
        runF32(
                TensorInternalAccess.float32Data(inputs.get(0)), inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.float32Data(inputs.get(1)), inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.float32Data(inputs.get(2)), inputs.get(2).getStorageOffsetUnsafe(),
                TensorInternalAccess.float32Data(node), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        maxGrad gradOp = (maxGrad) op;
        runBF16(
                TensorInternalAccess.bfloat16Data(inputs.get(0)), inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.bfloat16Data(inputs.get(1)), inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.bfloat16Data(inputs.get(2)), inputs.get(2).getStorageOffsetUnsafe(),
                TensorInternalAccess.bfloat16Data(node), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    static void runF64(double[] a, int aBaseOffset, double[] b, int bBaseOffset, double[] outGrad, int outGradBaseOffset, double[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runF64(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }

    static void runF32(float[] a, int aBaseOffset, float[] b, int bBaseOffset, float[] outGrad, int outGradBaseOffset, float[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runF32(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }

    static void runBF16(short[] a, int aBaseOffset, short[] b, int bBaseOffset, short[] outGrad, int outGradBaseOffset, short[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runBF16(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }
}
