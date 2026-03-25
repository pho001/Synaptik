package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.AddF16;
import Backend.kernels.cpu.f32.AddF32;
import Backend.kernels.cpu.f64.AddF64;
import Operations.Operation;
import Operations.add;
import Tensor.BroadcastPlan;
import Tensor.Tensor;

import java.util.List;

public class CpuAddKernel implements CpuKernel {
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
            BroadcastBinaryKernel.runF64(Operation.OpType.ADD, a, b, out, plan, mode, config);
            return;
        }
        AddF64.run(a, b, out, mode, config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] a = inputs.get(0).getFloat32Data();
        float[] b = inputs.get(1).getFloat32Data();
        float[] out = node.getFloat32Data();
        CpuExecutionMode mode = config.modeFor(op, node);
        ResolvedBroadcastPlan plan = resolvePlan(op, node);
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF32(Operation.OpType.ADD, a, b, out, plan, mode, config);
            return;
        }
        AddF32.run(a, b, out, mode, config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] a = inputs.get(0).getFloat16Data();
        short[] b = inputs.get(1).getFloat16Data();
        short[] out = node.getFloat16Data();
        CpuExecutionMode mode = config.modeFor(op, node);
        ResolvedBroadcastPlan plan = resolvePlan(op, node);
        if (plan != null && !plan.isNoBroadcast()) {
            BroadcastBinaryKernel.runF16(Operation.OpType.ADD, a, b, out, plan, mode, config);
            return;
        }
        AddF16.run(a, b, out, mode, config);
    }

    private static ResolvedBroadcastPlan resolvePlan(Operation op, Tensor node) {
        ResolvedBroadcastPlan resolved = node.getResolvedBroadcastPlan();
        if (resolved != null) {
            return resolved;
        }
        return ResolvedBroadcastPlan.from(extractPlan(op));
    }

    private static BroadcastPlan extractPlan(Operation op) {
        if (op instanceof add addOp) {
            return addOp.getBroadcastPlan();
        }
        return null;
    }
}
