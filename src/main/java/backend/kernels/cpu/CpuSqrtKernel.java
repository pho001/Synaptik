package backend.kernels.cpu;

import backend.kernels.cpu.bf16.UnaryBF16;
import backend.kernels.cpu.f32.UnaryF32;
import backend.kernels.cpu.f64.UnaryF64;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuSqrtKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF64.sqrt(inputs.get(0).getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        UnaryF32.sqrt(inputs.get(0).getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] continuation = context.inputFloatContinuation(0, node.getFlatDataSize());
        if (continuation != null) {
            UnaryBF16.sqrt(continuation, node.getBFloat16Data(), context.dispatchHints());
            return;
        }
        UnaryBF16.sqrt(inputs.get(0).getBFloat16Data(), node.getBFloat16Data(), context.dispatchHints());
    }
}
