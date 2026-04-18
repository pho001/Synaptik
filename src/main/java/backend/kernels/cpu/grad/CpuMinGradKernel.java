package backend.kernels.cpu.grad;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.elementwise.binary.minGrad;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public final class CpuMinGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runF64(
                inputs.get(0).getFloat64Data(), inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat64Data(), inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat64Data(), inputs.get(2).getStorageOffsetUnsafe(),
                node.getFloat64Data(), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runF32(
                inputs.get(0).getFloat32Data(), inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat32Data(), inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat32Data(), inputs.get(2).getStorageOffsetUnsafe(),
                node.getFloat32Data(), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runBF16(
                inputs.get(0).getBFloat16Data(), inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getBFloat16Data(), inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getBFloat16Data(), inputs.get(2).getStorageOffsetUnsafe(),
                node.getBFloat16Data(), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    static void runF64(double[] a, int aBaseOffset, double[] b, int bBaseOffset, double[] outGrad, int outGradBaseOffset, double[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runF64(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, false);
    }

    static void runF32(float[] a, int aBaseOffset, float[] b, int bBaseOffset, float[] outGrad, int outGradBaseOffset, float[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runF32(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, false);
    }

    static void runBF16(short[] a, int aBaseOffset, short[] b, int bBaseOffset, short[] outGrad, int outGradBaseOffset, short[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradExecutor.runBF16(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, false);
    }
}
