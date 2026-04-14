package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.unary.bf16.PowBF16;
import backend.kernels.cpu.elementwise.unary.f32.PowF32;
import backend.kernels.cpu.elementwise.unary.f64.PowF64;
import operations.Operation;
import operations.pow;
import tensor.Tensor;

import java.util.List;

public final class CpuPowKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        PowF64.run(inputs.get(0).getFloat64Data(), ((pow) op).getExponent(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        PowF32.run(inputs.get(0).getFloat32Data(), ((pow) op).getExponentF32(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        PowBF16.run(inputs.get(0).getBFloat16Data(), ((pow) op).getExponent(), node.getBFloat16Data(), context.dispatchHints());
    }
}
