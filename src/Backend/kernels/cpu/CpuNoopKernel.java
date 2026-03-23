package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuNoopKernel implements CpuKernel {
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
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }
}
