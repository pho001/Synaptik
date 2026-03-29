package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.UnaryF16;
import Backend.kernels.cpu.f32.UnaryF32;
import Backend.kernels.cpu.f64.UnaryF64;
import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuInvKernel implements CpuKernel {
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
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        UnaryF64.inv(in, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        UnaryF32.inv(in, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] in = inputs.get(0).getFloat16Data();
        short[] out = node.getFloat16Data();
        UnaryF16.inv(in, out, config.modeFor(op, node), config);
    }
}
