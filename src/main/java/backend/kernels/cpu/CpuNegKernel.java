package backend.kernels.cpu;

import backend.kernels.cpu.bf16.NegBF16;
import backend.kernels.cpu.f32.NegF32;
import backend.kernels.cpu.f64.NegF64;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuNegKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        NegF64.run(in, out, context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        NegF32.run(in, out, context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] in = inputs.get(0).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        NegBF16.run(in, out, context.dispatchHints());
    }
}
