package backend.kernels.cpu;

import backend.kernels.cpu.f16.UnaryF16;
import backend.kernels.cpu.f32.UnaryF32;
import backend.kernels.cpu.f64.UnaryF64;
import operations.Operation;
import operations.clampMax;
import tensor.Tensor;

import java.util.List;

public final class CpuClampMaxKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF64.clampMax(inputs.get(0).getFloat64Data(), ((clampMax) op).getMaxValue(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF32.clampMax(inputs.get(0).getFloat32Data(), ((clampMax) op).getMaxValueF32(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF16.clampMax(inputs.get(0).getFloat16Data(), ((clampMax) op).getMaxValueF32(), node.getFloat16Data(), context.dispatchHints());
    }
}
