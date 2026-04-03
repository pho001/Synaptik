package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalNotKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        BoolKernelSupport.runUnary(Operation.OpType.LOGICAL_NOT, inputs.get(0).getBoolData(), node.getBoolData());
    }
}
