package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorRemap;

import java.util.List;

public class CpuContiguousKernel implements CpuKernel {
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
        TensorRemap.apply(inputs.getFirst(), node, config.contiguousMaterializeThreshold());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        TensorRemap.apply(inputs.getFirst(), node, config.contiguousMaterializeThreshold());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        TensorRemap.apply(inputs.getFirst(), node, config.contiguousMaterializeThreshold());
    }
}
