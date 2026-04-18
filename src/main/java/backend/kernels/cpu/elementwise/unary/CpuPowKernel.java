package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuPowSupport;
import backend.kernels.cpu.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.unary.bf16.PowBF16;
import backend.kernels.cpu.elementwise.unary.f32.PowF32;
import backend.kernels.cpu.elementwise.unary.f64.PowF64;
import operations.Operation;
import operations.pow;
import tensor.Tensor;

import java.util.List;

public final class CpuPowKernel implements CpuKernel, ScalarUnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public double applyF64(double value, double parameter) {
        return CpuPowSupport.applyF64(value, parameter);
    }

    @Override
    public float applyF32(float value, float parameter) {
        return CpuPowSupport.applyF32(value, parameter);
    }

    @Override
    public float applyBF16(float value, float parameter) {
        return CpuPowSupport.applyF32(value, parameter);
    }

    @Override
    public boolean supportsDirectF64() {
        return true;
    }

    @Override
    public void runDirectF64(double[] in, double parameter, double[] out, ResolvedDispatchHints hints) {
        PowF64.run(in, parameter, out, hints);
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] in, float parameter, float[] out, ResolvedDispatchHints hints) {
        PowF32.run(in, parameter, out, hints);
    }

    @Override
    public boolean supportsDirectBF16() {
        return true;
    }

    @Override
    public void runDirectBF16(short[] in, float[] continuation, float parameter, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            PowBF16.run(continuation, parameter, out, hints);
        } else {
            PowBF16.run(in, parameter, out, hints);
        }
    }
}
