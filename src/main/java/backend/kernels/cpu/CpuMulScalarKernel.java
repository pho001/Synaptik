package backend.kernels.cpu;

import backend.kernels.cpu.bf16.MulScalarBF16;
import backend.kernels.cpu.f32.MulScalarF32;
import backend.kernels.cpu.f64.MulScalarF64;
import operations.Operation;
import operations.mulScalar;
import tensor.Tensor;

import java.util.List;

public class CpuMulScalarKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        double scalar = ((mulScalar) op).getScalar();
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        MulScalarF64.run(in, scalar, out, context.dispatchHints());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        MulScalarF32.run(in, ((mulScalar) op).getScalarF32(), out, context.dispatchHints());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        short[] in = inputs.get(0).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        MulScalarBF16.run(in, ((mulScalar) op).getScalar(), out, context.dispatchHints());
    }
}
