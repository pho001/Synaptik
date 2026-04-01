package backend.kernels.cpu;

import backend.kernels.cpu.f16.MulF16;
import backend.kernels.cpu.f32.MulF32;
import backend.kernels.cpu.f64.MulF64;
import operations.Operation;
import operations.mul;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public class CpuMulKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double[] a = inputs.get(0).getFloat64Data();
        double[] b = inputs.get(1).getFloat64Data();
        double[] out = node.getFloat64Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF64(Operation.OpType.MUL, a, b, out, plan, hints);
            return;
        }
        MulF64.run(a, b, out, hints);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] a = inputs.get(0).getFloat32Data();
        float[] b = inputs.get(1).getFloat32Data();
        float[] out = node.getFloat32Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF32(Operation.OpType.MUL, a, b, out, plan, hints);
            return;
        }
        MulF32.run(a, b, out, hints);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] a = inputs.get(0).getFloat16Data();
        short[] b = inputs.get(1).getFloat16Data();
        short[] out = node.getFloat16Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF16(Operation.OpType.MUL, a, b, out, plan, hints);
            return;
        }
        MulF16.run(a, b, out, hints);
    }
}
