package backend.kernels.cpu;

import backend.kernels.cpu.f16.UnaryF16;
import backend.kernels.cpu.f32.UnaryF32;
import backend.kernels.cpu.f64.UnaryF64;
import operations.Operation;
import operations.clampMin;
import tensor.Tensor;

import java.util.List;

public final class CpuClampMinKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF64.clampMin(inputs.get(0).getFloat64Data(), ((clampMin) op).getMinValue(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF32.clampMin(inputs.get(0).getFloat32Data(), ((clampMin) op).getMinValueF32(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF16.clampMin(inputs.get(0).getFloat16Data(), ((clampMin) op).getMinValueF32(), node.getFloat16Data(), context.dispatchHints());
    }
}
