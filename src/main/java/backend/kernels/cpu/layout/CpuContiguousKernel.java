package backend.kernels.cpu.layout;

import backend.kernels.cpu.*;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorRemap;

import java.util.List;

public class CpuContiguousKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        TensorRemap.apply(inputs.getFirst(), node, context.planner().contiguousMaterializeThreshold());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        TensorRemap.apply(inputs.getFirst(), node, context.planner().contiguousMaterializeThreshold());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        TensorRemap.apply(inputs.getFirst(), node, context.planner().contiguousMaterializeThreshold());
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        TensorRemap.apply(inputs.getFirst(), node, context.planner().contiguousMaterializeThreshold());
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        TensorRemap.apply(inputs.getFirst(), node, context.planner().contiguousMaterializeThreshold());
    }
}
