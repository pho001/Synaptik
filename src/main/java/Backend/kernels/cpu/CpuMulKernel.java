package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.MulF16;
import Backend.kernels.cpu.f32.MulF32;
import Backend.kernels.cpu.f64.MulF64;
import Operations.Operation;
import Operations.mul;
import Tensor.BroadcastPlan;
import Tensor.Tensor;

import java.util.List;

public class CpuMulKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forwardF64(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double[] a = inputs.get(0).getFloat64Data();
        double[] b = inputs.get(1).getFloat64Data();
        double[] out = node.getFloat64Data();
        CpuExecutionMode mode = config.modeFor(op, node);
        ResolvedBroadcastPlan plan = resolvePlan(op, node);
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF64(Operation.OpType.MUL, a, b, out, plan, mode, config);
            return;
        }
        MulF64.run(a, b, out, mode, config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] a = inputs.get(0).getFloat32Data();
        float[] b = inputs.get(1).getFloat32Data();
        float[] out = node.getFloat32Data();
        CpuExecutionMode mode = config.modeFor(op, node);
        ResolvedBroadcastPlan plan = resolvePlan(op, node);
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF32(Operation.OpType.MUL, a, b, out, plan, mode, config);
            return;
        }
        MulF32.run(a, b, out, mode, config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] a = inputs.get(0).getFloat16Data();
        short[] b = inputs.get(1).getFloat16Data();
        short[] out = node.getFloat16Data();
        CpuExecutionMode mode = config.modeFor(op, node);
        ResolvedBroadcastPlan plan = resolvePlan(op, node);
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF16(Operation.OpType.MUL, a, b, out, plan, mode, config);
            return;
        }
        MulF16.run(a, b, out, mode, config);
    }

    private static ResolvedBroadcastPlan resolvePlan(Operation op, Tensor node) {
        ResolvedBroadcastPlan resolved = node.getResolvedBroadcastPlan();
        if (resolved != null) {
            return resolved;
        }
        return ResolvedBroadcastPlan.from(extractPlan(op));
    }

    private static BroadcastPlan extractPlan(Operation op) {
        if (op instanceof mul mulOp) {
            return mulOp.getBroadcastPlan();
        }
        return null;
    }
}
