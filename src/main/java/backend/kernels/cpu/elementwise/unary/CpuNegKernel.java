package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.unary.bf16.NegBF16;
import backend.kernels.cpu.elementwise.unary.f32.NegF32;
import backend.kernels.cpu.elementwise.unary.f64.NegF64;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuNegKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        NegF64.run(inputs.get(0).getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        NegF32.run(inputs.get(0).getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        NegBF16.run(inputs.get(0).getBFloat16Data(), node.getBFloat16Data(), context.dispatchHints());
    }
}
