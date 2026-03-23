package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.PowF16;
import Backend.kernels.cpu.f32.PowF32;
import Backend.kernels.cpu.f64.PowF64;
import Operations.Operation;
import Operations.pow;
import Tensor.Tensor;

import java.util.List;

public class CpuPowKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forwardF64(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double exponent = ((pow) op).getExponent();
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        PowF64.run(in, exponent, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        PowF32.run(in, ((pow) op).getExponent(), out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] in = inputs.get(0).getFloat16Data();
        short[] out = node.getFloat16Data();
        PowF16.run(in, ((pow) op).getExponent(), out, config.modeFor(op, node), config);
    }
}
