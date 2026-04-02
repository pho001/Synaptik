package backend.kernels.cpu;

import operations.Operation;
import operations.minGrad;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public final class CpuMinGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runF64(inputs.get(0).getFloat64Data(), inputs.get(1).getFloat64Data(), inputs.get(2).getFloat64Data(),
                node.getFloat64Data(), gradOp.getBroadcastPlan(), gradOp.isForFirstInput());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runF32(inputs.get(0).getFloat32Data(), inputs.get(1).getFloat32Data(), inputs.get(2).getFloat32Data(),
                node.getFloat32Data(), gradOp.getBroadcastPlan(), gradOp.isForFirstInput());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        minGrad gradOp = (minGrad) op;
        runF16(inputs.get(0).getFloat16Data(), inputs.get(1).getFloat16Data(), inputs.get(2).getFloat16Data(),
                node.getFloat16Data(), gradOp.getBroadcastPlan(), gradOp.isForFirstInput());
    }

    static void runF64(double[] a, double[] b, double[] outGrad, double[] out, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF64(a, b, outGrad, out, plan, forFirstInput, false);
    }

    static void runF32(float[] a, float[] b, float[] outGrad, float[] out, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF32(a, b, outGrad, out, plan, forFirstInput, false);
    }

    static void runF16(short[] a, short[] b, short[] outGrad, short[] out, BroadcastPlan plan, boolean forFirstInput) {
        MinMaxGradKernelSupport.runF16(a, b, outGrad, out, plan, forFirstInput, false);
    }
}
