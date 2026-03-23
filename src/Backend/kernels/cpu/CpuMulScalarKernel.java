package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.MulScalarF16;
import Backend.kernels.cpu.f32.MulScalarF32;
import Backend.kernels.cpu.f64.MulScalarF64;
import Operations.Operation;
import Operations.mulScalar;
import Tensor.Tensor;

import java.util.List;

public class CpuMulScalarKernel implements CpuKernel {
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
        double scalar = ((mulScalar) op).getScalar();
        double[] in = inputs.get(0).getFloat64Data();
        double[] out = node.getFloat64Data();
        MulScalarF64.run(in, scalar, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] in = inputs.get(0).getFloat32Data();
        float[] out = node.getFloat32Data();
        MulScalarF32.run(in, (float) ((mulScalar) op).getScalar(), out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] in = inputs.get(0).getFloat16Data();
        short[] out = node.getFloat16Data();
        MulScalarF16.run(in, ((mulScalar) op).getScalar(), out, config.modeFor(op, node), config);
    }
}
