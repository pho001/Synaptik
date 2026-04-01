package backend.kernels.cpu;

import backend.kernels.cpu.f16.UnaryF16;
import backend.kernels.cpu.f32.UnaryF32;
import backend.kernels.cpu.f64.UnaryF64;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuLogKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF64.log(inputs.get(0).getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF32.log(inputs.get(0).getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF16.log(inputs.get(0).getFloat16Data(), node.getFloat16Data(), context.dispatchHints());
    }
}
