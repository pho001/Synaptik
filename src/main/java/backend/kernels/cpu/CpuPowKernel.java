package backend.kernels.cpu;

import backend.kernels.cpu.bf16.PowBF16;
import backend.kernels.cpu.f32.PowF32;
import backend.kernels.cpu.f64.PowF64;
import operations.Operation;
import operations.pow;
import tensor.Tensor;

import java.util.List;

public class CpuPowKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double exponent = ((pow) op).getExponent();
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        PowF64.run(in, exponent, out, context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        PowF32.run(in, ((pow) op).getExponentF32(), out, context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] in = inputs.get(0).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        PowBF16.run(in, ((pow) op).getExponent(), out, context.dispatchHints());
    }
}
