package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalOrKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        BoolKernelSupport.runBinary(Operation.OpType.LOGICAL_OR, inputs.get(0).getBoolData(), inputs.get(1).getBoolData(), node.getBoolData(), context.broadcastPlan());
    }
}
