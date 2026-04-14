package backend.kernels.cpu.elementwise.binary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.binary.bf16.MulBF16;
import backend.kernels.cpu.elementwise.binary.f32.MulF32;
import backend.kernels.cpu.elementwise.binary.f64.MulF64;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuMulKernel implements CpuKernel, BinaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double[] a = inputs.get(0).getFloat64Data();
        double[] b = inputs.get(1).getFloat64Data();
        double[] out = node.getFloat64Data();
        ResolvedDispatchHints hints = context.dispatchHints();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        if (plan != null && !plan.isNoBroadcast()) {
            ElementwiseBinaryExecutor.execute(this, inputs, node, context);
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
            ElementwiseBinaryExecutor.execute(this, inputs, node, context);
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
            ElementwiseBinaryExecutor.execute(this, inputs, node, context);
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

    @Override
    public double applyF64(double left, double right) {
        return left * right;
    }

    @Override
    public float applyF32(float left, float right) {
        return left * right;
    }

    @Override
    public float applyBF16(float left, float right) {
        return left * right;
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) {
        return left.mul(right);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector left, FloatVector right) {
        return left.mul(right);
    }
}
