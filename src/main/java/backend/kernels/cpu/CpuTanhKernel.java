package backend.kernels.cpu;

import backend.kernels.cpu.f16.UnaryF16;
import backend.kernels.cpu.f32.UnaryF32;
import backend.kernels.cpu.f64.UnaryF64;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuTanhKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (context.useFastTanhApprox()) {
            UnaryF64.fastTanh(inputs.get(0).getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
        } else {
            UnaryF64.tanh(inputs.get(0).getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
        }
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (context.useFastTanhApprox()) {
            UnaryF32.fastTanh(inputs.get(0).getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
        } else {
            UnaryF32.tanh(inputs.get(0).getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
        }
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (context.useFastTanhApprox()) {
            UnaryF16.fastTanh(inputs.get(0).getFloat16Data(), node.getFloat16Data(), context.dispatchHints());
        } else {
            UnaryF16.tanh(inputs.get(0).getFloat16Data(), node.getFloat16Data(), context.dispatchHints());
        }
    }
}
