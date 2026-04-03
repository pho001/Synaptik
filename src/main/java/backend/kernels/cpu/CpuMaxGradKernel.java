package backend.kernels.cpu;

import operations.Operation;
import operations.maxGrad;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        maxGrad gradOp = (maxGrad) op;
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
        maxGrad gradOp = (maxGrad) op;
        runF32(
                inputs.get(0).getFloat32Data(), inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat32Data(), inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat32Data(), inputs.get(2).getStorageOffsetUnsafe(),
                node.getFloat32Data(), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        maxGrad gradOp = (maxGrad) op;
        runF16(
                inputs.get(0).getFloat16Data(), inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat16Data(), inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat16Data(), inputs.get(2).getStorageOffsetUnsafe(),
                node.getFloat16Data(), node.getStorageOffsetUnsafe(),
                gradOp.getBroadcastPlan(), gradOp.isForFirstInput()
        );
    }

    static void runF64(double[] a, int aBaseOffset, double[] b, int bBaseOffset, double[] outGrad, int outGradBaseOffset, double[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF64(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }

    static void runF32(float[] a, int aBaseOffset, float[] b, int bBaseOffset, float[] outGrad, int outGradBaseOffset, float[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF32(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }

    static void runF16(short[] a, int aBaseOffset, short[] b, int bBaseOffset, short[] outGrad, int outGradBaseOffset, short[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF16(a, aBaseOffset, b, bBaseOffset, outGrad, outGradBaseOffset, out, outBaseOffset, plan, forFirstInput, true);
    }
}
