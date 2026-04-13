package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import backend.kernels.cpu.bf16.MulBF16;
import backend.kernels.cpu.f32.MulF32;
import backend.kernels.cpu.f64.MulF64;
import operations.Operation;
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
            ElementwiseBinaryExecutor.execute(NumericBinaryOp.MUL, inputs, node, context);
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
            ElementwiseBinaryExecutor.execute(NumericBinaryOp.MUL, inputs, node, context);
            return;
        }
        MulF32.run(a, b, out, hints);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] a = inputs.get(0).getBFloat16Data();
        short[] b = inputs.get(1).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            ElementwiseBinaryExecutor.execute(NumericBinaryOp.MUL, inputs, node, context);
            return;
        }
        float[] ac = context.inputFloatContinuation(0, node.getFlatDataSize());
        float[] bc = context.inputFloatContinuation(1, node.getFlatDataSize());
        if (ac != null && bc != null) {
            MulBF16.run(ac, bc, out, hints);
            return;
        }
        if (ac != null) {
            MulBF16.run(ac, b, out, hints);
            return;
        }
        if (bc != null) {
            MulBF16.run(a, bc, out, hints);
            return;
        }
        MulBF16.run(a, b, out, hints);
    }
}
