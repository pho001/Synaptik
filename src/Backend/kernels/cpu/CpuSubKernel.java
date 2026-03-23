package Backend.kernels.cpu;

import Backend.kernels.cpu.f16.SubF16;
import Backend.kernels.cpu.f32.SubF32;
import Backend.kernels.cpu.f64.SubF64;
import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuSubKernel implements CpuKernel {
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
        double[] a = inputs.get(0).getFloat64Data();
        double[] b = inputs.get(1).getFloat64Data();
        double[] out = node.getFloat64Data();
        SubF64.run(a, b, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        float[] a = inputs.get(0).getFloat32Data();
        float[] b = inputs.get(1).getFloat32Data();
        float[] out = node.getFloat32Data();
        SubF32.run(a, b, out, config.modeFor(op, node), config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        short[] a = inputs.get(0).getFloat16Data();
        short[] b = inputs.get(1).getFloat16Data();
        short[] out = node.getFloat16Data();
        SubF16.run(a, b, out, config.modeFor(op, node), config);
    }
}
