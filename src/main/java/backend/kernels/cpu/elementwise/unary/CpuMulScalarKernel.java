package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.unary.bf16.MulScalarBF16;
import backend.kernels.cpu.elementwise.unary.f32.MulScalarF32;
import backend.kernels.cpu.elementwise.unary.f64.MulScalarF64;
import operations.Operation;
import operations.mulScalar;
import tensor.Tensor;

import java.util.List;

public final class CpuMulScalarKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MulScalarF64.run(inputs.get(0).getFloat64Data(), ((mulScalar) op).getScalar(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MulScalarF32.run(inputs.get(0).getFloat32Data(), ((mulScalar) op).getScalarF32(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MulScalarBF16.run(inputs.get(0).getBFloat16Data(), ((mulScalar) op).getScalar(), node.getBFloat16Data(), context.dispatchHints());
    }
}
