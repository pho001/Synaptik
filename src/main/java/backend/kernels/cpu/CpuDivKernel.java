package backend.kernels.cpu;

import backend.kernels.cpu.bf16.DivBF16;
import backend.kernels.cpu.f32.DivF32;
import backend.kernels.cpu.f64.DivF64;
import operations.Operation;
import operations.div;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.List;

public class CpuDivKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double[] a = inputs.get(0).getFloat64Data();
        double[] b = inputs.get(1).getFloat64Data();
        double[] out = node.getFloat64Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF64(Operation.OpType.DIV, a, b, out, plan, hints);
            return;
        }
        DivF64.run(a, b, out, hints);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] a = inputs.get(0).getFloat32Data();
        float[] b = inputs.get(1).getFloat32Data();
        float[] out = node.getFloat32Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF32(Operation.OpType.DIV, a, b, out, plan, hints);
            return;
        }
        DivF32.run(a, b, out, hints);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] a = inputs.get(0).getBFloat16Data();
        short[] b = inputs.get(1).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runBF16(Operation.OpType.DIV, a, b, out, plan, hints);
            return;
        }
        DivBF16.run(a, b, out, hints);
    }
}
